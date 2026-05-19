import os
import json
import tarfile
import logging
import datetime
import shutil
from pathlib import Path
from confluent_kafka import Producer as KafkaProducer

from kafka_consumer import make_consumer, consume_loop, send_kafka_message
from minio_client import make_client, download, upload
from export import ExportPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

minio = make_client()

_EXPORT_BUCKET = "exports"

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


def _ensure_bucket(client, bucket: str):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info(f"Bucket '{bucket}' creato")


def handle_job(job: dict):
    """
    Pipeline di export CPP:
    1. Download modello quantizzato (.aidge) da MinIO
    2. Download test split
    3. Generazione ed compilazione export C++
    4. Upload archivio tar.gz su MinIO
    5. Notifica topic done-jobs
    """
    job_id = job["job_id"]
    model_minio_path = job["export_path"]
    split_key = job.get("test_split") or job["train_split"]
    label_column = job.get("label_column")

    base_dir = Path(f"/app/downloads/{job_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Inizio export CPP Job: {job_id}")
        send_progress(job_id, "stage", "export", "Avvio export C++")

        # --- 1. DOWNLOAD MODELLO QUANTIZZATO ---
        model_local = str(base_dir / "model_quant.aidge")
        download(minio, "quantized-models", model_minio_path, model_local)
        logger.info(f"Modello quantizzato scaricato: {model_local}")
        send_progress(job_id, "log", "export", "Modello quantizzato scaricato")

        # --- 2. DOWNLOAD SPLIT ---
        split_local = str(base_dir / "data.csv")
        download(minio, "datasets", split_key, split_local)
        logger.info(f"Split scaricato: datasets/{split_key}")

        # --- 3. EXPORT CPP ---
        send_progress(job_id, "log", "export", "Generazione codice C++ in corso...")
        export_folder = base_dir / "export_cpp"
        pipeline = ExportPipeline(
            model_path=str(base_dir / "model_quant.aidge"),
            test_csv=split_local,
            label=label_column,
        )
        pipeline.run(export_folder)
        logger.info(f"Export CPP completato: {export_folder}")
        send_progress(job_id, "log", "export", "Codice C++ generato. Compilazione in corso...")

        # --- 4. ARCHIVIO E UPLOAD ---
        _ensure_bucket(minio, _EXPORT_BUCKET)
        tar_local = str(base_dir / f"export_{job_id}.tar.gz")
        with tarfile.open(tar_local, "w:gz") as tar:
            tar.add(export_folder, arcname="export_cpp")

        minio_dest = f"{job_id}/export_cpp.tar.gz"
        upload(minio, _EXPORT_BUCKET, minio_dest, tar_local)
        logger.info(f"Archivio caricato su MinIO: {_EXPORT_BUCKET}/{minio_dest}")

        # --- 5. NOTIFICA ---
        send_kafka_message(
            os.environ["KAFKA_TOPIC_OUT"],
            {"job_id": job_id, "export_archive": minio_dest},
        )
        logger.info(f"Notifica export inviata per Job {job_id}")
        send_progress(
            job_id, "done", "done",
            "Pipeline completata con successo!",
            data={"export_archive": minio_dest},
        )

    except Exception as e:
        logger.error(f"Errore critico nel Job {job_id}: {str(e)}", exc_info=True)
        send_progress(job_id, "error", "export", f"Errore: {str(e)}")

    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info(f"🧹 Pulizia cartella temporanea completata per {job_id}")


consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"],
    topic=os.environ["KAFKA_TOPIC_IN"],
    group_id=os.environ["KAFKA_GROUP_ID"],
)

consume_loop(consumer, handle_job)
