import os
import sys
import json
import importlib.util
import logging
import tarfile
import torch
import shutil
import datetime
import hashlib
import pandas as pd
from pathlib import Path
from confluent_kafka import Producer as KafkaProducer
from kafka_consumer import make_consumer, consume_loop, send_kafka_message
from minio_client import make_client, download, upload
from train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MinIO client once, reusable across all jobs
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


def load_model_class(script_path: str, class_name: str):
    """Dynamically load the model class from the downloaded script."""
    spec = importlib.util.spec_from_file_location("custom_model", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def handle_job(job: dict):
    """
    Main logic for processing a training job:
    1. Setup working directories
    2. Download assets from MinIO
    3. Dynamic model instantiation
    4. Load weights and train
    5. Save and upload results
    6. Cleanup
    """
    job_id          = job["job_id"]
    class_name      = job["class_name"]
    # Parameters for the model constructor (e.g. input_channels, etc.)
    model_params    = job.get("model_params", {})

    # Local working path inside the container
    base_dir = Path(f"/app/downloads/{job_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting job: {job_id}")
        send_progress(job_id, "stage", "training", f"Starting training — model: {class_name}")

        # --- 1. DOWNLOAD FROM MINIO ---
        script_path = download(minio, "models", job["model_script"], str(base_dir / "model_script.py"))

        dataset_local = str(base_dir / job["dataset"])
        download(minio, "datasets", job["dataset"], dataset_local)

        header = pd.read_csv(dataset_local, nrows=0).columns
        label = job.get("label_column") or header[-1]
        feature_count = len([c for c in header if c != label])
        if model_params.get("input_channels") != feature_count:
            raise ValueError(f"input_channels must be {feature_count} for the selected input columns")

        # --- 2. MODEL INSTANTIATION ---
        ModelClass = load_model_class(str(script_path), class_name)
        model = ModelClass(**model_params)
        logger.info(f"Model {class_name} instantiated successfully.")

        # --- 3. TRAINING ---
        def _epoch_callback(event):
            if event["type"] == "epoch":
                send_progress(
                    job_id, "epoch", "training",
                    f"Epoch {event['epoch']}/{event['max_epochs']} | "
                    f"Train: {event['train_loss']:.4f} | Val: {event['val_loss']:.4f} | "
                    f"no_improve: {event['no_improve']}/{event['patience']}",
                    data={k: v for k, v in event.items() if k != "type"},
                )
            else:
                send_progress(job_id, "log", "training", event["message"])

        model, test_metrics, split_dfs, deployment = train(
            model,
            dataset_local,
            patience=job.get("patience", 10),
            max_epochs=job.get("max_epochs", 500),
            label_column=job.get("label_column"),
            train_ratio=job.get("train_ratio", 0.70),
            val_ratio=job.get("val_ratio", 0.15),
            progress_callback=_epoch_callback,
        )
        logger.info(f"Test metrics: {test_metrics}")
        send_progress(
            job_id, "stage", "quantization",
            "Training complete. Starting quantization...",
            data={"metrics": test_metrics},
        )

        # --- 5. UPLOAD SPLIT CSV TO MINIO ---
        splits_dir = base_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        for split_name, split_df in split_dfs.items():
            split_local = splits_dir / f"{split_name}.csv"
            split_df.to_csv(str(split_local), index=False)
            minio_split_key = f"{job_id}/{split_name}.csv"
            upload(minio, "datasets", minio_split_key, str(split_local))
            logger.info(f"Split '{split_name}' uploaded to datasets/{minio_split_key}")

        deployment["job_id"] = job_id
        deployment["dataset_sha256"] = hashlib.sha256(Path(dataset_local).read_bytes()).hexdigest()
        deployment_local = base_dir / "deployment.json"
        deployment_local.write_text(json.dumps(deployment, allow_nan=False))
        upload(minio, "datasets", f"{job_id}/deployment.json", str(deployment_local))

        # --- 6. ONNX EXPORT ---
        import onnx as _onnx
        onnx_path = base_dir / "model.onnx"
        torch.manual_seed(int(os.environ.get("TRAINING_SEED", "42")))
        if model_params["input_channels"] != len(deployment["feature_names"]):
            raise ValueError("input_channels must match the number of input features")
        dummy_input = torch.randn(1, model_params["input_channels"], 1)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True,
        )
        # Force inline save: remove external .data file if generated
        _onnx_model = _onnx.load(str(onnx_path))
        _onnx.save_model(_onnx_model, str(onnx_path), save_as_external_data=False)
        logger.info(f"ONNX exported (inline, no external .data file): {onnx_path}")

        # --- 7. SAVE WEIGHTS ---
        output_path = base_dir / "trained_model.pth"
        torch.save(model.state_dict(), str(output_path))

        # --- 8. UPLOAD MODEL ---
        minio_pth_dest  = f"{job_id}/{class_name}_trained.pth"
        minio_onnx_dest = f"{job_id}/{class_name}_final.onnx"

        upload(minio, "models", minio_pth_dest,  str(output_path))
        upload(minio, "onnx",   minio_onnx_dest, str(onnx_path))

        logger.info(f".pth uploaded to: {minio_pth_dest}")
        logger.info(f".onnx uploaded to: {minio_onnx_dest}")

        quantizer_request = {
            "job_id": job_id,
            "onnx_path": minio_onnx_dest,
            "train_split": f"{job_id}/train.csv",
            "test_split": f"{job_id}/test.csv",
            "label_column": job.get("label_column"),
            "test_metrics": test_metrics,
            "deployment": f"{job_id}/deployment.json",
        }
        send_kafka_message(quantizer_request)
        logger.info(f"Quantization request sent for job {job_id}")

    except Exception as e:
        logger.error(f"Critical error in job {job_id}: {str(e)}", exc_info=True)
        send_progress(job_id, "error", "training", f"Error: {str(e)}")

    finally:
        # --- 9. FINAL CLEANUP ---
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info(f"Temporary folder cleaned for {job_id}")


consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"].strip().strip('"'),
    topic=os.environ["KAFKA_TOPIC_IN"].strip().strip('"'),
    group_id=os.environ["KAFKA_GROUP_ID"].strip().strip('"'),
)

consume_loop(consumer, handle_job)
