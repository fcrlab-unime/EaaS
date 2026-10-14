import os
import json
import tarfile
import logging
import datetime
import shutil
import subprocess
import re
import time
from pathlib import Path
from confluent_kafka import Producer as KafkaProducer

from kafka_consumer import make_consumer, consume_loop, send_kafka_message
from minio_client import make_client, download
from uart_bridge import UARTBridge, start_uart_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

minio = make_client()

IDF_TARGET  = os.environ.get("IDF_TARGET",  "esp32")
FLASH_PORT  = os.environ.get("FLASH_PORT",  "/dev/ttyUSB0")
FLASH_BAUD  = os.environ.get("FLASH_BAUD",  "460800")
IDF_PATH    = os.environ.get("IDF_PATH",    "/opt/esp/idf")

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
        "job_id":    job_id,
        "type":      type_,
        "stage":     stage,
        "message":   message,
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


# ── IDF project creation ────────────────────────────────────────────────────

def _fix_aidge_sources(idf_dir: Path):
    """Strip Aidge-specific section attributes incompatible with ESP32 linker."""
    targets = (
        list((idf_dir / "main" / "data").rglob("*.h")) +
        list((idf_dir / "components" / "dnn" / "src" / "parameters").rglob("*.cpp")) +
        list((idf_dir / "components" / "dnn" / "include" / "parameters").rglob("*.hpp"))
    )
    for f in targets:
        text = f.read_text()
        cleaned = re.sub(r'\s*__attribute__\s*\(\s*\(\s*section\s*\([^)]*\)\s*\)\s*\)', '', text)
        cleaned = re.sub(r'#include\s*<complex>\s*\n', '', cleaned)
        cleaned = re.sub(r'using\s+namespace\s+std::complex_literals\s*;\s*\n', '', cleaned)
        if cleaned != text:
            f.write_text(cleaned)


def _create_idf_project(idf_dir: Path, export_dir: Path):
    main_dir      = idf_dir / "main"
    dnn_component = idf_dir / "components" / "dnn"
    main_dir.mkdir(parents=True)
    dnn_component.mkdir(parents=True)

    # Top-level CMakeLists.txt
    (idf_dir / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "include($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
        "project(dnn_inference)\n"
    )

    # sdkconfig.defaults
    (idf_dir / "sdkconfig.defaults").write_text(
        "CONFIG_COMPILER_CXX_EXCEPTIONS=y\n"
        "CONFIG_COMPILER_CXX_RTTI=y\n"
        "CONFIG_ESP_CONSOLE_NONE=y\n"
    )

    # Copy DNN component files
    shutil.copytree(export_dir / "dnn" / "include", dnn_component / "include")
    shutil.copytree(export_dir / "dnn" / "src",     dnn_component / "src")

    # Copy test data
    data_src = export_dir / "data"
    if data_src.exists():
        shutil.copytree(data_src, main_dir / "data")

    # Patch all aidge sources after copy
    _fix_aidge_sources(idf_dir)

    # DNN component CMakeLists.txt
    cpp_files = sorted((dnn_component / "src").rglob("*.cpp"))
    rel_cpp   = [str(f.relative_to(dnn_component)) for f in cpp_files]
    src_lines = "\n        ".join(f'"{f}"' for f in rel_cpp)
    (dnn_component / "CMakeLists.txt").write_text(
        "idf_component_register(\n"
        f"    SRCS\n"
        f"        {src_lines}\n"
        "    INCLUDE_DIRS \"include\"\n"
        ")\n"
        "\n"
        "target_compile_options(${COMPONENT_LIB} PRIVATE -std=c++17)\n"
    )

    metadata = json.loads((export_dir / "deployment.json").read_text())
    if metadata.get("output_format") != "soh_float":
        raise ValueError("Re-export model with SOH dequantization and scaler metadata")
    count = len(metadata["feature_names"])
    if not count or len(metadata["scale"]) != count or len(metadata["offset"]) != count:
        raise ValueError("Invalid scaler dimensions")
    import math
    if not all(math.isfinite(v) for v in metadata["scale"] + metadata["offset"]):
        raise ValueError("Non-finite scaler")
    settings = {
        "UART_PORT": int(os.getenv("INFERENCE_UART_PORT", "0")),
        "UART_BAUD": int(os.getenv("INFERENCE_UART_BAUD", "115200")),
        "UART_TX_GPIO": int(os.getenv("INFERENCE_UART_TX_GPIO", "1")),
        "UART_RX_GPIO": int(os.getenv("INFERENCE_UART_RX_GPIO", "3")),
        "LED_RGB_GPIO": int(os.getenv("LED_RGB_GPIO", "5")),
        "LED_BRIGHTNESS": int(os.getenv("LED_BRIGHTNESS", "32")),
    }
    led_pin = settings["LED_RGB_GPIO"]
    if led_pin not in {*range(0, 20), *range(21, 24), *range(25, 28), 32, 33} or 6 <= led_pin <= 11:
        raise ValueError("WS2812 GPIO must be a valid ESP32 output, excluding flash pins")
    if led_pin in {settings["UART_TX_GPIO"], settings["UART_RX_GPIO"]}:
        raise ValueError("WS2812 GPIO must not overlap UART")
    if not 1 <= settings["LED_BRIGHTNESS"] <= 255:
        raise ValueError("LED_BRIGHTNESS must be between 1 and 255")
    templates = Path(__file__).parent / "templates"
    for name in ("main.cpp", "protocol.hpp", "led.hpp"):
        shutil.copyfile(templates / name, main_dir / name)
    shutil.copyfile(export_dir / "model_adapter.hpp", main_dir / "model_adapter.hpp")
    header = "#pragma once\n#include <cstddef>\n"
    header += f"constexpr size_t FEATURE_COUNT = {count};\n"
    for name, value in settings.items():
        header += f"constexpr int {name} = {value};\n"
    for name, key in (("SCALE", "scale"), ("OFFSET", "offset")):
        header += f"constexpr double {name}[] = {{" + ",".join(repr(v) for v in metadata[key]) + "};\n"
    header += f"constexpr double INPUT_MULTIPLIER = {float(metadata['input_multiplier'])};\n"
    header += "constexpr char JOB_ID[] = " + json.dumps(metadata["job_id"]) + ";\n"
    (main_dir / "deployment.hpp").write_text(header)

    # main/CMakeLists.txt
    include_dirs = '"." "data"' if (main_dir / "data").exists() else '"."'
    (main_dir / "CMakeLists.txt").write_text(
        "idf_component_register(\n"
        "    SRCS \"main.cpp\"\n"
        f"    INCLUDE_DIRS {include_dirs}\n"
        "    REQUIRES dnn esp_driver_uart esp_driver_rmt\n"
        ")\n"
        "\n"
        "target_compile_options(${COMPONENT_LIB} PRIVATE -std=c++17)\n"
    )


# ── Serial monitor ──────────────────────────────────────────────────────────

SERIAL_READ_SECS = int(os.environ.get("SERIAL_READ_SECS", "8"))


def _read_serial_output(port: str, baud: int = 115200, duration: int = SERIAL_READ_SECS) -> str:
    """Read serial output from the device for `duration` seconds.

    Opens the port immediately after flash and polls with a short timeout
    so any data buffered by the kernel (or arriving shortly after) is
    captured without wasting time on sleeps.
    """
    try:
        import serial
    except ImportError:
        return "(pyserial not available)"
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
        lines = []
        deadline = time.time() + duration
        while time.time() < deadline:
            line = ser.readline()
            if line:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                lines.append(decoded)
                logger.info("[SERIAL] %s", decoded)
        ser.close()
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Serial read failed: %s", e)
        return f"(serial read error: {e})"


# ── Shell helper ────────────────────────────────────────────────────────────

def _run_shell(cmd: str, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    full_cmd = f". {IDF_PATH}/export.sh 2>/dev/null && {cmd}"
    return subprocess.run(
        full_cmd,
        shell=True,
        executable="/bin/bash",
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "IDF_PATH": IDF_PATH},
    )


# ── Job handler ─────────────────────────────────────────────────────────────

def _handle_job(job: dict):
    job_id       = job["job_id"]
    archive_path = job["export_archive"]

    base_dir = Path(f"/tmp/flash_{job_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        send_progress(job_id, "stage", "flash", "Starting flash service")

        # 1. Download archive
        tar_local = str(base_dir / "export_cpp.tar.gz")
        send_progress(job_id, "log", "flash", "Downloading export archive from MinIO...")
        download(minio, "exports", archive_path, tar_local)
        logger.info("Archive downloaded: %s", tar_local)

        # 2. Extract
        with tarfile.open(tar_local, "r:gz") as tar:
            tar.extractall(base_dir)
        export_dir = base_dir / "export_cpp"
        logger.info("Archive extracted: %s", export_dir)

        # 3. Create IDF project
        send_progress(job_id, "log", "flash", "Creating ESP-IDF project...")
        idf_dir = base_dir / "idf_project"
        _create_idf_project(idf_dir, export_dir)
        logger.info("IDF project created: %s", idf_dir)

        # 4. Build
        send_progress(job_id, "log", "flash", f"Building for target {IDF_TARGET} (this may take a few minutes)...")
        build_cmd = f"idf.py set-target {IDF_TARGET} && idf.py build"
        result = _run_shell(build_cmd, idf_dir, timeout=600)
        if result.returncode != 0:
            logger.error("Build failed:\n%s\n%s", result.stdout[-3000:], result.stderr[-3000:])
            raise RuntimeError(f"idf.py build failed (rc={result.returncode})")
        logger.info("Build complete")
        send_progress(job_id, "log", "flash", "Build complete. Flashing...")

        # 5. Flash
        flash_cmd = f"idf.py flash -p {FLASH_PORT} -b {FLASH_BAUD}"
        result = _run_shell(flash_cmd, idf_dir, timeout=120)
        if result.returncode != 0:
            logger.error("Flash failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
            raise RuntimeError(f"idf.py flash failed (rc={result.returncode}): {result.stderr[-500:]}")
        logger.info("Flash complete on %s", FLASH_PORT)

        # 6. Read serial output — apri la porta subito, PRIMA di qualsiasi chiamata Kafka
        serial_out = _read_serial_output(FLASH_PORT)
        logger.info("Serial output:\n%s", serial_out)
        send_progress(job_id, "log", "flash", f"Reading serial output ({SERIAL_READ_SECS}s)...")

        # 7. Done
        send_kafka_message(
            os.environ["KAFKA_TOPIC_OUT"],
            {"job_id": job_id, "status": "flashed", "port": FLASH_PORT, "serial_output": serial_out},
        )
        send_progress(
            job_id, "done", "done",
            f"Firmware flashed successfully on {FLASH_PORT}!",
            data={"port": FLASH_PORT, "target": IDF_TARGET, "serial_output": serial_out},
        )
        logger.info("Job %s completed", job_id)

    except Exception as e:
        logger.error("Critical error in job %s: %s", job_id, e, exc_info=True)
        send_progress(job_id, "error", "flash", f"Flash error: {e}")

    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info("Temporary folder cleaned for %s", job_id)


uart_bridge = UARTBridge(
    os.getenv("INFERENCE_SERIAL_PORT", FLASH_PORT),
    int(os.getenv("INFERENCE_UART_BAUD", "115200")),
)
start_uart_server(uart_bridge, int(os.getenv("UART_API_PORT", "8081")))


def handle_job(job: dict):
    with uart_bridge.flash_session():
        _handle_job(job)


consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"].strip().strip('"'),
    topic=os.environ["KAFKA_TOPIC_IN"].strip().strip('"'),
    group_id=os.environ["KAFKA_GROUP_ID"].strip().strip('"'),
)

consume_loop(consumer, handle_job)
