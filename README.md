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
| Export | `export/` | `export-jobs` | `export-done` | Generates standalone C++ inference code from the quantized model |
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
   export backend, archives the generated project as a `.tar.gz`, uploads it, and publishes a job to `export-done`.
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
   subclass — see `sample/CCN1D.py` for an example 1D CNN) and pick the detected
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

## Continuous EIS inference and RGB LED

New training jobs export `datasets/<job_id>/deployment.json`: the exact feature
order, fitted MinMaxScaler coefficients (`scale` and `offset`), split seed,
dataset hash and raw held-out test examples with their true SOH. Quantization
adds its input/output contract; export includes this metadata and a generated
`model_adapter.hpp` in the archive.

For `sample/df_clean.csv`, select `soh` as target and set `input_channels`
to the number of remaining columns. All columns except the target are used as
inputs; there is no column-exclusion setting. To omit an identifier such as
`cell`, remove it from the CSV before uploading it. A CSV with 60 interleaved
real/imaginary EIS pairs plus `temperature` has 121 inputs. Frequency positions
must match the training dataset. Retrain the model whenever input columns change.

The ESP-IDF application is maintained in `flash/templates/main.cpp`, rather than
patching Aidge's desktop smoke-test main. It continuously accepts newline-ended
CSV frames of raw features over UART, with exactly the exported feature count.
Each feature is transformed as:

```text
normalized[i] = raw[i] * scale[i] + offset[i]
input_int8[i] = round_to_even(clamp(normalized[i] * 127, -127, 127))
```

Aidge is configured with `rescale_inputs=False` and `dequantize=True`: the graph
reconstructs the scalar SOH in its original target units. It is not divided by
127 in the frontend. The target is expected to be SOH in percent (0–100).
Predicted SOH **≤ 80** lights red; **> 80** lights green. LEDs are off before the
first valid inference and after an invalid frame. The last valid colour remains
until the next frame. Invalid counts, non-finite values and oversized frames are
rejected; quantization saturation is reported in the response.

UART defaults to UART0, TX GPIO1/RX GPIO3, 115200 baud, 8N1, through the FireBeetle's
USB-to-serial bridge. The application console is disabled to keep application
logs out of the protocol; ROM/bootloader messages can still appear at reset.
Send `HELLO\n` to get the firmware job ID and feature count. Responses are JSON
lines, for example:

```json
{"type":"inference","job_id":"a1b2c3d4","soh":78.3,"color":"red","clipped_features":0}
```

The DFRobot FireBeetle ESP32-E V1.0 (DFR0654) has an onboard WS2812 RGB LED
on GPIO5/D8. The firmware drives this LED through ESP-IDF's RMT peripheral,
sending one GRB pixel followed by a 300 us reset pulse; no external LED or
additional downloaded component is needed. See the
[DFRobot onboard RGB documentation](https://wiki.dfrobot.com/dfr0654/docs/23253).

The default active-channel brightness is 32 out of 255. Copy the desired
`LED_RGB_GPIO=5` and `LED_BRIGHTNESS=32` settings from
`flash/firmware.env.example` into `flash/.env` before building/flashing a new job.
These settings are compiled into the firmware. The previous external-LED
settings (`LED_RED_GPIO`, `LED_GREEN_GPIO`, `LED_BLUE_GPIO`, `LED_ACTIVE_LOW`)
are no longer used and can be removed from `flash/.env`.

After updating the UART panel/bridge, rebuild and restart only their services:

```bash
docker compose up -d --build --no-deps frontend flash-service
```

This does not rebuild Aidge or change the firmware. An already flashed UART
firmware can be used immediately. In the frontend panel **Battery health check**:

1. Wait for flashing to finish, close other serial monitors, then click
   **Connect ESP32**. The server opens the USB port on the Docker host, reads
   the firmware job ID and loads that job's raw test examples.
2. Choose a curve and click **Check battery health**. Its input values are
   filled in automatically.
3. Use **Repeat all curves** to cycle through the test examples; **Stop** ends
   the replay. Compare the estimated SOH with the known battery health.
4. Under **Use a custom curve**, upload a TXT or CSV file containing one curve,
   or paste comma-separated original measurements into **Curve values**.
   Headered CSV columns are reordered to the model's input order. Without a
   header, values must already be in that order. Click **Check battery health**
   to send the curve. The target is not required; every input column is required.
   You can also load samples by model ID or download the deployment metadata.

The browser uses the frontend HTTP API, which forwards requests to the UART
bridge inside the flash service. No Web Serial/browser permission, HTTPS or
special browser is required for UART access: the ESP32 must be attached to the
Docker host. The bridge API is internal to the Docker network (port 8081 is not
published on the host). It validates firmware job ID and feature count, rejects
malformed curves, and invalidates the connection on timeout or board reset.
The flash service closes the bridge connection before each flash job and blocks
UART requests until the job, including serial capture, finishes. Reconnect from
the panel afterwards.

By default the bridge uses `FLASH_PORT` and `INFERENCE_UART_BAUD` (115200).
`INFERENCE_SERIAL_PORT` can select another mapped device. If you customize the
internal API port with `UART_API_PORT`, set `FLASH_SERVICE_URL` in the frontend
environment accordingly (default `http://flash-service:8081`).

Static reference frames generated from the default deterministic split are in
[examples/uart](examples/uart/README.md); the frontend's job-specific samples are
authoritative.
