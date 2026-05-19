import asyncio
import io
import json
import logging
import os
import uuid
from collections import defaultdict

from confluent_kafka import Consumer as KafkaConsumer
from confluent_kafka import Producer
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from minio import Minio
from minio.error import S3Error

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EaaS Frontend")

# ---------------------------------------------------------------------------
# Per-job progress state (in-memory)
# ---------------------------------------------------------------------------
_job_histories: dict = defaultdict(list)
_job_queues: dict = defaultdict(list)   # job_id -> list[asyncio.Queue]


async def _kafka_progress_listener():
    """Background task: consumes job-logs topic and fans out to SSE queues."""
    loop = asyncio.get_event_loop()

    def _make_consumer():
        c = KafkaConsumer({
            "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "redpanda:29092").strip().strip('"'),
            "group.id": f"frontend-progress-{uuid.uuid4().hex[:8]}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })
        c.subscribe([os.environ.get("KAFKA_TOPIC_LOGS", "job-logs")])
        return c

    consumer = await loop.run_in_executor(None, _make_consumer)
    logger.info("Progress Kafka listener avviato")

    try:
        while True:
            msg = await loop.run_in_executor(None, lambda: consumer.poll(0.5))
            if msg is None or msg.error():
                continue
            try:
                data = json.loads(msg.value().decode("utf-8"))
                job_id = data.get("job_id")
                if not job_id:
                    continue
                _job_histories[job_id].append(data)
                for q in list(_job_queues.get(job_id, [])):
                    await q.put(data)
            except Exception as e:
                logger.warning("Progress listener parse error: %s", e)
    finally:
        consumer.close()


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_kafka_progress_listener())


# ---------------------------------------------------------------------------
# MinIO / Kafka helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

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


@app.get("/api/progress/{job_id}")
async def job_progress(job_id: str, request: Request):
    """SSE endpoint: stream job progress events to the browser."""
    queue: asyncio.Queue = asyncio.Queue()
    _job_queues[job_id].append(queue)

    async def event_generator():
        # Replay past events for clients that connect late
        for event in list(_job_histories.get(job_id, [])):
            yield f"data: {json.dumps(event)}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep-alive
        finally:
            queues = _job_queues.get(job_id, [])
            if queue in queues:
                queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
