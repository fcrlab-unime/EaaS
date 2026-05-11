import os
import sys
import json
import logging
import pandas as pd
import shutil
from pathlib import Path
from kafka_consumer import make_consumer, consume_loop, send_kafka_message
from minio_client import make_client, download, upload
from quant import PipelineConfig
import aidge_core.aidge_save

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

minio = make_client()


def handle_job(job: dict):
    """
    Pipeline di quantizzazione:
    1. Download ONNX da MinIO
    2. Download dataset
    3. Quantizzazione con Aidge
    4. Upload modello quantizzato
    5. Notifica topic exporting_jobs
    """
    job_id          = job["job_id"]
    train_split_key = job["train_split"]
    onnx_minio_path = job["onnx_path"]
    label_column    = job.get("label_column")

    base_dir = Path(f"/app/downloads/{job_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"⚙️  Inizio quantizzazione Job: {job_id}")

        # --- 1. DOWNLOAD ONNX DA MINIO ---
        onnx_local = str(base_dir / "model.onnx")
        download(minio, "onnx", onnx_minio_path, onnx_local)
        logger.info(f"ONNX scaricato: {onnx_local}")

        # --- 2. DOWNLOAD TRAIN SPLIT DEL JOB ---
        train_local = str(base_dir / "train.csv")
        download(minio, "datasets", train_split_key, train_local)
        logger.info(f"Train split scaricato: datasets/{train_split_key}")

        dataset_df = pd.read_csv(train_local)
        logger.info(f"Train split shape: {dataset_df.shape}")

        # --- 3. QUANTIZZAZIONE ---
        cfg = PipelineConfig(
            model_path=onnx_local,
            train_csv=train_local,
            label=label_column,
        )
        
        quant_model = cfg.quant()

        aidge_core.aidge_save.save_model(quant_model, str(base_dir / "model_quant"))
        
        # --- 4. UPLOAD MODELLO QUANTIZZATO ---
        minio_export_dest = f"quantized_models/{job_id}/model_quant.aidge"
        upload(minio, "quantized-models", minio_export_dest, str(base_dir / "model_quant.aidge"))
        logger.info(f"Modello quantizzato caricato su MinIO: {minio_export_dest}")
        
        export_request = {
            "job_id":      job_id,
            "export_path": minio_export_dest,
            "train_split": train_split_key,
            "test_split":  job.get("test_split"),
            "label_column": label_column,
        }
        send_kafka_message(os.environ["KAFKA_TOPIC_OUT"], export_request)
        logger.info(f"Richiesta export inviata per Job {job_id}")

    except Exception as e:
        logger.error(f"Errore critico nel Job {job_id}: {str(e)}", exc_info=True)

    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info(f"🧹 Pulizia cartella temporanea completata per {job_id}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"],
    topic=os.environ["KAFKA_TOPIC_IN"],
    group_id=os.environ["KAFKA_GROUP_ID"],
)

consume_loop(consumer, handle_job)