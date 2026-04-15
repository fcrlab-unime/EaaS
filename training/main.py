import os
import sys
import json
import importlib.util
import logging
import tarfile
import torch
from pathlib import Path
from kafka_consumer import make_consumer, consume_loop
from minio_client import make_client, download, upload
from train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializza client MinIO una sola volta, riutilizzabile per tutti i job
minio = make_client()


def load_model_class(script_path: str, class_name: str):
    """Carica dinamicamente la classe del modello dallo script scaricato."""
    spec = importlib.util.spec_from_file_location("custom_model", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def handle_job(job: dict):
    job_id    = job["job_id"]
    class_name = job["class_name"]
    base_dir  = Path(f"/app/downloads/{job_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download da MinIO
    script_path  = download(minio, "models",   job["model_script"], str(base_dir / "model_script.py"))
    weights_path = download(minio, "weights",  job["weights"],       str(base_dir / "weights.pth"))
    dataset_path = download(minio, "datasets", job["dataset"],       str(base_dir / "dataset.zip"))

    # 2. Estrai dataset
    dataset_dir = base_dir / "data"
    dataset_dir.mkdir(exist_ok=True)
    with tarfile.open(dataset_path) as tar:
        tar.extractall(dataset_dir)

    # 3. Istanzia modello e carica weights
    ModelClass = load_model_class(str(script_path), class_name)
    model = ModelClass()
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))

    # 4. Training
    model = train(model, dataset_dir, epochs=job["epochs"])

    # 5. Salva modello con metadati
    output_path = base_dir / "trained.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "class_name": class_name,
        "job_id":     job_id,
        "epochs":     job["epochs"],
        "model_script": job["model_script"],
    }, str(output_path))

    # 6. Upload su MinIO
    upload(minio, "models", f"trained/{job_id}/{class_name}.pth", str(output_path))
    logger.info("Job %s completed, model uploaded.", job_id)

consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"],
    topic=os.environ["KAFKA_TOPIC_IN"],
    group_id=os.environ["KAFKA_GROUP_ID"],
)

consume_loop(consumer, handle_job)