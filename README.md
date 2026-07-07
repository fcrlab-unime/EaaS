# EaaS — Embedded (AI) as a Service

EaaS is an end-to-end pipeline that turns a PyTorch model and a CSV dataset into
firmware running on a real microcontroller. Submit a model script and a
dataset through the web UI and the pipeline will automatically:

1. **Train** the model on your data.
2. **Quantize** it to int8.
3. **Export** it to standalone C++ inference code.
4. **Flash** the compiled firmware onto an ESP32 board.

Every stage is a separate microservice that reacts to events on a Kafka
(Redpanda) bus and exchanges files through MinIO (S3-compatible object
storage). Progress and logs from all stages are streamed live to the browser.

## Architecture

```
┌──────────┐   train-jobs    ┌──────────┐  quant-jobs   ┌──────────────┐  export-jobs  ┌──────────┐  export-done  ┌──────────┐
│ Frontend │ ──────────────▶ │ Training │ ────────────▶ │ Quantization │ ────────────▶ │  Export  │ ────────────▶ │  Flash   │
│ (FastAPI)│                 │ service  │               │   service    │               │ service  │               │ service  │
└──────────┘ ◀────────────── └──────────┘ ◀──────────── └──────────────┘ ◀──────────── └──────────┘ ◀──────────── └──────────┘
      ▲                              job-logs (progress events, consumed by the frontend via SSE)
      │
      └── every service also reads/writes intermediate artifacts to MinIO buckets
```

* **Redpanda** (Kafka-compatible) is the message bus. Each stage consumes one
  topic, does its work, and produces a message on the next topic once done.
* **MinIO** stores every artifact that needs to move between stages: uploaded
  scripts/datasets, dataset splits, `.pth` weights, ONNX models, quantized
  `.aidge` models, and the generated C++/firmware archive.
* Every service also publishes progress/log events to the `job-logs` topic;
  the frontend consumes this topic and streams it to the browser over
  Server-Sent Events (SSE), driving the step-by-step progress UI.

### Services

| Service | Folder | Consumes | Produces | Purpose |
|---|---|---|---|---|
| Frontend | `frontend/` | — | `train-jobs` | Web UI: upload model script + dataset, submit a job, watch live progress |
| Training | `training/` | `train-jobs` | `quant-jobs` | Trains the model, splits the dataset, exports ONNX |
| Quantization | `quantization/` | `quant-jobs` | `export-jobs` | int8-quantizes the ONNX model with Aidge, using the training set for calibration |
| Export | `export/` | `export-jobs` | `export-done` | Generates and compiles standalone C++ inference code from the quantized model |
| Flash | `flash/` | `export-done` | `flash-done` | Wraps the C++ export into an ESP-IDF project, builds it, and flashes it to an ESP32 over serial |

All topic names, MinIO credentials and other per-service settings are
configured through each service's `.env` file (see `training/.env`,
`quantization/.env`, `export/.env`, `flash/.env`, `frontend/.env`).

### MinIO buckets

| Bucket | Content |
|---|---|
| `models` | Uploaded model scripts (`.py`) and trained weights (`.pth`) |
| `datasets` | Uploaded datasets and the train/val/test splits produced by training |
| `onnx` | ONNX export of the trained (float) model |
| `quantized-models` | int8-quantized model (`.aidge` format) |
| `exports` | Generated C++ project, packaged as a `.tar.gz` archive |
| `bin` | Reserved for compiled binaries |

## Pipeline walkthrough

1. **Submit** — the user uploads a model script (a `torch.nn.Module`
   subclass), a CSV dataset, picks the label column, the model's constructor
   parameters, and early-stopping settings (`patience`, `max_epochs`) via the
   frontend. The frontend uploads both files to MinIO and publishes a job to
   `train-jobs`.
2. **Training** (`training/main.py`, `training/train.py`) — downloads the
   script and dataset, dynamically instantiates the model class, trains it
   with early stopping on a stratified train/val/test split (features scaled
   with `MinMaxScaler`), evaluates it on the test set (MAE/MSE/RMSE/R²),
   exports it to ONNX, uploads weights + ONNX + splits to MinIO, and
   publishes a job to `quant-jobs`.
3. **Quantization** (`quantization/main.py`, `quantization/quant.py`) —
   downloads the ONNX model and the train/test splits, quantizes the model
   to int8 using [Aidge](export/aidge), calibrating on a sample of the
   training set, recomputes metrics on the test set to report the accuracy
   impact of quantization, uploads the `.aidge` model, and publishes a job to
   `export-jobs`.
4. **Export** (`export/main.py`, `export/export.py`) — downloads the
   quantized model, generates standalone C++ inference code via Aidge's C++
   export backend, compiles and smoke-tests the binary, archives the result
   as a `.tar.gz`, uploads it, and publishes a job to `export-done`.
5. **Flash** (`flash/main.py`) — downloads and extracts the C++ export,
   wraps it into an ESP-IDF project (patching Aidge-generated sources for
   ESP32 compatibility), builds it with `idf.py build`, flashes it to the
   board over serial with `idf.py flash`, reads back the device's serial
   output, and publishes the final `flash-done` event (also surfaced to the
   UI as the "done" step, including the captured serial output).

Throughout every stage, `send_progress()` publishes structured events
(`stage`, `epoch`, `log`, `error`, `done`) to `job-logs`, which the frontend
relays to the browser to drive the stepper UI, the live training-loss chart,
and the metrics panels (float vs. quantized).

## Prerequisites

* Docker and Docker Compose
* An ESP32 board connected over USB/serial (only required for the `flash`
  stage — `/dev/ttyUSB0` by default, configurable via `flash/.env`)

## Running the pipeline

From the project root, build and start every service (MinIO, Redpanda,
training, quantization, export, flash, frontend) with the top-level compose
file, which includes all the per-service compose files:

```bash
docker compose up --build
```

Once everything is up:

* **Frontend UI**: http://localhost:8080
* **MinIO console**: http://localhost:9001 (default credentials `minioadmin` /
  `minioadmin`, see `minio/.env`)
* **Redpanda** Kafka API: `localhost:9092`

To use it:

1. Open the frontend at http://localhost:8080.
2. Upload your model script (a `.py` file defining a `torch.nn.Module`
   subclass — see `CCN1D.py` for an example 1D CNN) and pick the detected
   class name.
3. Fill in the constructor parameters (auto-detected from the class
   signature when possible).
4. Upload your dataset CSV and pick the target/label column.
5. Set `patience` and `max_epochs`, then click **Start**.
6. Watch the pipeline progress live through the Training → Quantization →
   Export → Flash → Done steps, including live training loss and pre/post
   quantization metrics.

If you only have the ESP32 attached to a different machine than the one
running Docker, only the `flash` container needs access to the serial port
(`devices: /dev/ttyUSB0` in `flash/flash_compose.yml`) — adjust `FLASH_PORT`
in `flash/.env` accordingly.

## Configuration reference

Every service reads its settings from an `.env` file in its own folder
(`training/.env`, `quantization/.env`, `export/.env`, `flash/.env`,
`frontend/.env`, `minio/.env`). The most relevant variables:

* `MINIO_ENDPOINT`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` — MinIO
  connection/credentials.
* `KAFKA_BOOTSTRAP` — Redpanda bootstrap address (`redpanda:29092` inside
  the Docker network).
* `KAFKA_TOPIC_IN` / `KAFKA_TOPIC_OUT` / `KAFKA_GROUP_ID` — per-stage topic
  wiring (matches the table above).
* `KAFKA_TOPIC_LOGS` — shared progress/log topic (`job-logs`).
* `TRAINING_SEED` / `QUANT_SEED` — reproducibility seeds for training and
  quantization.
* `QUANT_CALIB_SAMPLES` — number of calibration samples used for int8
  quantization.
* `IDF_TARGET`, `FLASH_PORT`, `FLASH_BAUD`, `IDF_PATH`, `SERIAL_READ_SECS` —
  ESP-IDF build/flash target and serial settings (`flash/.env`).
