import asyncio
import csv
import io
import json
import logging
import os
import uuid
import urllib.request
import urllib.error
from collections import defaultdict

from confluent_kafka import Consumer as KafkaConsumer
from confluent_kafka import Producer
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from minio import Minio
from minio.error import S3Error
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = FastAPI(title="EaaS Frontend")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
    return Producer({"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "redpanda:29092")})


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
):
    try:
        params = json.loads(model_params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="model_params non è un JSON valido")

    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="model_params must be a JSON object")

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
        columns = next(csv.reader(io.StringIO(dataset_bytes.decode("utf-8-sig"))), [])
        if label_column not in columns:
            raise HTTPException(400, "Select a target column present in the dataset")
        features = [c for c in columns if c != label_column]
        if params.get("input_channels") != len(features):
            raise HTTPException(400, f"input_channels must be {len(features)} for the selected features")
        _put(minio, "datasets", dataset.filename, dataset_bytes, "text/csv")
        logger.info("Dataset caricato: datasets/%s", dataset.filename)

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
    producer = _kafka_producer()
    producer.produce(
        topic=os.environ.get("KAFKA_TOPIC_IN", "train-jobs"),
        value=json.dumps(job).encode("utf-8"),
        callback=lambda err, msg: logger.error("Kafka delivery failed: %s", err) if err else logger.info("Job %s inviato a Kafka", job_id),
    )
    producer.flush()

    logger.info("Job %s sottomesso con successo", job_id)
    return {"job_id": job_id, "status": "submitted"}


@app.get("/api/download/{job_id}")
async def download_export(job_id: str):
    """Download the C++ export archive for a given job."""
    minio = _minio()
    object_name = f"{job_id}/export_cpp.tar.gz"
    try:
        obj = minio.get_object("exports", object_name)
        data = obj.read()
        obj.close()
        obj.release_conn()
        return Response(
            content=data,
            media_type="application/gzip",
            headers={"Content-Disposition": f'attachment; filename="export_{job_id}.tar.gz"'},
        )
    except S3Error as e:
        raise HTTPException(status_code=404, detail=f"Export archive not found: {e}")


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


@app.get("/api/deployment/{job_id}")
def deployment_metadata(job_id: str):
    """Raw held-out examples and scaler from this exact training job."""
    try:
        obj = _minio().get_object("datasets", f"{job_id}/deployment.json")
        try:
            return json.loads(obj.read())
        finally:
            obj.close()
            obj.release_conn()
    except S3Error:
        raise HTTPException(404, "Deployment metadata unavailable: run a new training job")


@app.api_route("/api/uart/{action}", methods=["GET", "POST"])
async def uart_proxy(action: str, request: Request):
    """The board is attached to the flash host, not the browser's computer."""
    expected_method = "GET" if action == "status" else "POST"
    if action not in {"status", "connect", "disconnect", "send"} or request.method != expected_method:
        raise HTTPException(404, "Endpoint UART non trovato")
    data = None
    if request.method == "POST":
        data = await request.body()
        if len(data) > 131072:
            raise HTTPException(400, "Richiesta troppo grande")
    base = os.getenv("FLASH_SERVICE_URL", "http://flash-service:8081").rstrip("/")

    def forward():
        upstream = urllib.request.Request(f"{base}/{action}", data=data,
                                          headers={"Content-Type": "application/json"}, method=request.method)
        try:
            with urllib.request.urlopen(upstream, timeout=15) as response:
                return Response(response.read(), status_code=response.status, media_type="application/json")
        except urllib.error.HTTPError as error:
            return Response(error.read(), status_code=error.code, media_type="application/json")
        except (urllib.error.URLError, TimeoutError):
            raise HTTPException(503, "Servizio UART non raggiungibile: aggiorna e avvia il servizio flash")
    return await asyncio.to_thread(forward)
