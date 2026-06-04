import os
import sys
import json
import logging
import datetime
import pandas as pd
import shutil
from pathlib import Path
from confluent_kafka import Producer as KafkaProducer
from kafka_consumer import make_consumer, consume_loop, send_kafka_message
from minio_client import make_client, download, upload
from quant import PipelineConfig
import aidge_core.aidge_save

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

minio = make_client()

_progress_producer = None

def _get_progress_producer():
    global _progress_producer
    if _progress_producer is None:
        _progress_producer = KafkaProducer({
            "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "redpanda:29092").strip().strip('"'),
        })
    return _progress_producer

def send_progress(job_id: str, type_: str, stage: str, message: str, data: dict = None):
    payload = {
        "job_id": job_id,
        "type": type_,
        "stage": stage,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if data:
        payload["data"] = data
    try:
        p = _get_progress_producer()
        p.produce(
            topic=os.environ.get("KAFKA_TOPIC_LOGS", "job-logs"),
            value=json.dumps(payload).encode("utf-8"),
        )
        p.poll(0)
    except Exception as e:
        logger.warning("send_progress failed: %s", e)


def handle_job(job: dict):
    """
    Quantization pipeline:
    1. Download ONNX from MinIO
    2. Download dataset
    3. Quantize with Aidge
    4. Upload quantized model
    5. Notify exporting_jobs topic
    """
    job_id          = job["job_id"]
    train_split_key = job["train_split"]
    onnx_minio_path = job["onnx_path"]
    label_column    = job.get("label_column")

    base_dir = Path(f"/app/downloads/{job_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting quantization job: {job_id}")
        send_progress(job_id, "stage", "quantization", "Starting ONNX quantization")

        # --- 1. DOWNLOAD ONNX FROM MINIO ---
        onnx_local = str(base_dir / "model.onnx")
        download(minio, "onnx", onnx_minio_path, onnx_local)
        logger.info(f"ONNX downloaded: {onnx_local}")
        send_progress(job_id, "log", "quantization", "ONNX model downloaded")

        # --- 2. DOWNLOAD TRAIN SPLIT ---
        train_local = str(base_dir / "train.csv")
        download(minio, "datasets", train_split_key, train_local)
        logger.info(f"Train split downloaded: datasets/{train_split_key}")

        dataset_df = pd.read_csv(train_local)
        logger.info(f"Train split shape: {dataset_df.shape}")
        send_progress(job_id, "log", "quantization", f"Calibration dataset: {dataset_df.shape[0]} samples, {dataset_df.shape[1]} columns")

        # --- 3. QUANTIZATION ---
        send_progress(job_id, "log", "quantization", "Quantization in progress...")
        cfg = PipelineConfig(
            model_path=onnx_local,
            train_csv=train_local,
            label=label_column,
        )

        quant_model = cfg.quant()

        aidge_core.aidge_save.save_model(quant_model, str(base_dir / "model_quant"))
        send_progress(job_id, "log", "quantization", "Quantization complete")

        # --- 3b. POST-QUANTIZATION METRICS ---
        quant_metrics = None
        test_split_key = job.get("test_split")
        if test_split_key:
            send_progress(job_id, "log", "quantization", "Computing post-quantization metrics on test set...")
            try:
                test_local = str(base_dir / "test.csv")
                download(minio, "datasets", test_split_key, test_local)
                quant_metrics = cfg.compute_metrics(quant_model, test_local)
                logger.info(f"Post-quant metrics: {quant_metrics}")
                send_progress(
                    job_id, "log", "quantization",
                    f"Post-Quant → MAE: {quant_metrics['mae']:.4f} | "
                    f"RMSE: {quant_metrics['rmse']:.4f} | R²: {quant_metrics['r2']:.4f}",
                )
            except Exception as exc:
                logger.warning("Post-quant metrics computation failed: %s", exc)
                send_progress(job_id, "log", "quantization",
                              f"Post-quant metrics unavailable: {exc}")

        # --- 4. UPLOAD QUANTIZED MODEL ---
        minio_export_dest = f"quantized_models/{job_id}/model_quant.aidge"
        upload(minio, "quantized-models", minio_export_dest, str(base_dir / "model_quant.aidge"))
        logger.info(f"Quantized model uploaded to MinIO: {minio_export_dest}")

        export_request = {
            "job_id":      job_id,
            "export_path": minio_export_dest,
            "train_split": train_split_key,
            "test_split":  test_split_key,
            "label_column": label_column,
        }
        send_kafka_message(os.environ["KAFKA_TOPIC_OUT"], export_request)
        logger.info(f"Export request sent for job {job_id}")
        send_progress(
            job_id, "stage", "export",
            "Quantization complete. Starting C++ export...",
            data={"quant_metrics": quant_metrics} if quant_metrics else None,
        )

    except Exception as e:
        logger.error(f"Critical error in job {job_id}: {str(e)}", exc_info=True)
        send_progress(job_id, "error", "quantization", f"Error: {str(e)}")

    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info(f"Temporary folder cleaned for {job_id}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"].strip().strip('"'),
    topic=os.environ["KAFKA_TOPIC_IN"].strip().strip('"'),
    group_id=os.environ["KAFKA_GROUP_ID"].strip().strip('"'),
)

consume_loop(consumer, handle_job)
