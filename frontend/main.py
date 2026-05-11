import io
import json
import logging
import os
import uuid

from confluent_kafka import Producer
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from minio import Minio
from minio.error import S3Error

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EaaS Frontend")


def _minio() -> Minio:
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def _put(client: Minio, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    client.put_object(bucket, key, io.BytesIO(data), len(data), content_type=content_type)


def _kafka_producer() -> Producer:
    return Producer({"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")})


@app.post("/api/submit")
async def submit_job(
    model_script: UploadFile = File(...),
    class_name: str = Form(...),
    model_params: str = Form(...),
    dataset: UploadFile = File(...),
    label_column: str = Form(...),
    patience: int = Form(10),
    max_epochs: int = Form(500),
    weights: UploadFile = File(None),
):
    try:
        params = json.loads(model_params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="model_params non è un JSON valido")

    if patience < 1:
        raise HTTPException(status_code=400, detail="patience deve essere almeno 1")
    if max_epochs < 1:
        raise HTTPException(status_code=400, detail="max_epochs deve essere almeno 1")

    job_id = uuid.uuid4().hex[:8]
    minio = _minio()

    try:
        script_bytes = await model_script.read()
        _put(minio, "models", model_script.filename, script_bytes, "text/x-python")
        logger.info("Script caricato: models/%s", model_script.filename)

        dataset_bytes = await dataset.read()
        _put(minio, "datasets", dataset.filename, dataset_bytes, "text/csv")
        logger.info("Dataset caricato: datasets/%s", dataset.filename)

        weights_key = None
        if weights and weights.filename:
            weights_bytes = await weights.read()
            if weights_bytes:
                weights_key = weights.filename
                _put(minio, "weights", weights_key, weights_bytes)
                logger.info("Pesi caricati: weights/%s", weights_key)

    except S3Error as e:
        logger.error("Errore MinIO: %s", e)
        raise HTTPException(status_code=500, detail=f"Errore upload MinIO: {e}")

    job = {
        "job_id": job_id,
        "model_script": model_script.filename,
        "class_name": class_name,
        "model_params": params,
        "dataset": dataset.filename,
        "label_column": label_column,
        "patience": patience,
        "max_epochs": max_epochs,
    }
    if weights_key:
        job["weights"] = weights_key

    producer = _kafka_producer()
    producer.produce(
        topic=os.environ.get("KAFKA_TOPIC_IN", "train-jobs"),
        value=json.dumps(job).encode("utf-8"),
        callback=lambda err, msg: logger.error("Kafka delivery failed: %s", err) if err else logger.info("Job %s inviato a Kafka", job_id),
    )
    producer.flush()

    logger.info("Job %s sottomesso con successo", job_id)
    return {"job_id": job_id, "status": "submitted"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
