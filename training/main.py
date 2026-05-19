import os
import sys
import json
import importlib.util
import logging
import tarfile
import torch
import shutil
import datetime
from pathlib import Path
from confluent_kafka import Producer as KafkaProducer
from kafka_consumer import make_consumer, consume_loop, send_kafka_message
from minio_client import make_client, download, upload
from train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializza client MinIO una sola volta, riutilizzabile per tutti i job
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
    """Carica dinamicamente la classe del modello dallo script scaricato."""
    spec = importlib.util.spec_from_file_location("custom_model", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def handle_job(job: dict):
    """
    Logica principale per processare un job di training:
    1. Setup cartelle
    2. Download asset da MinIO
    3. Istanziazione dinamica del modello
    4. Caricamento pesi e Training
    5. Salvataggio e Upload risultati
    6. Cleanup
    """
    job_id = job["job_id"]
    class_name = job["class_name"]
    # Parametri per il costruttore della classe (es. input_channels, ecc.)
    model_params = job.get("model_params", {}) 
    
    # Percorso di lavoro locale al container
    base_dir = Path(f"/app/downloads/{job_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Inizio elaborazione Job: {job_id}")
        send_progress(job_id, "stage", "training", f"Avvio training — modello: {class_name}")

        # --- 1. DOWNLOAD DA MINIO ---
        script_path = download(minio, "models", job["model_script"], str(base_dir / "model_script.py"))

        dataset_local = str(base_dir / job["dataset"])
        download(minio, "datasets", job["dataset"], dataset_local)

        weights_path = None
        if job.get("weights"):
            weights_path = download(minio, "weights", job["weights"], str(base_dir / "weights.pth"))

        # --- 2. ISTANZIAZIONE MODELLO ---
        ModelClass = load_model_class(str(script_path), class_name)
        model = ModelClass(**model_params)
        logger.info(f"Modello {class_name} istanziato con successo.")

        # --- 3. CARICAMENTO PESI INIZIALI ---
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
            logger.info(f"Pesi iniziali caricati da {job['weights']}")
            send_progress(job_id, "log", "training", f"Pesi pre-addestrati caricati da {job['weights']}")

        # --- 4. TRAINING ---
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

        model, test_metrics, split_dfs = train(
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
            "Training completato. Avvio quantizzazione...",
            data={"metrics": test_metrics},
        )

        # --- 5. UPLOAD SPLIT CSV SU MINIO ---
        splits_dir = base_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        for split_name, split_df in split_dfs.items():
            split_local = splits_dir / f"{split_name}.csv"
            split_df.to_csv(str(split_local), index=False)
            minio_split_key = f"{job_id}/{split_name}.csv"
            upload(minio, "datasets", minio_split_key, str(split_local))
            logger.info(f"Split '{split_name}' caricato in splits/{minio_split_key}")

        # --- 7. EXPORT ONNX ---
        import onnx as _onnx
        onnx_path = base_dir / "model.onnx"
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
        # Forza salvataggio inline: elimina il file .data esterno se generato
        _onnx_model = _onnx.load(str(onnx_path))
        _onnx.save_model(_onnx_model, str(onnx_path), save_as_external_data=False)
        logger.info(f"ONNX esportato (inline, nessun .data esterno): {onnx_path}")

        # --- 8. SALVATAGGIO PESI ---
        output_path = base_dir / "trained_model.pth"
        torch.save(model.state_dict(), str(output_path))

        # --- 9. UPLOAD MODELLO ---
        minio_pth_dest  = f"{job_id}/{class_name}_trained.pth"
        minio_onnx_dest = f"{job_id}/{class_name}_final.onnx"

        upload(minio, "models", minio_pth_dest,  str(output_path))
        upload(minio, "onnx",   minio_onnx_dest, str(onnx_path))

        logger.info(f".pth caricato in: {minio_pth_dest}")
        logger.info(f".onnx caricato in: {minio_onnx_dest}")

        quantizer_request = {
            "job_id": job_id,
            "onnx_path": minio_onnx_dest,
            "train_split": f"{job_id}/train.csv",
            "test_split": f"{job_id}/test.csv",
            "label_column": job.get("label_column"),
            "test_metrics": test_metrics,
        }
        send_kafka_message(quantizer_request)
        logger.info(f"Richiesta di quantizzazione inviata per Job {job_id}")

    except Exception as e:
        logger.error(f"Errore critico nel Job {job_id}: {str(e)}", exc_info=True)
        send_progress(job_id, "error", "training", f"Errore: {str(e)}")

    finally:
        # --- 8. CLEANUP FINALE ---
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info(f"Pulizia cartella temporanea completata per {job_id}")


    


consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"],
    topic=os.environ["KAFKA_TOPIC_IN"],
    group_id=os.environ["KAFKA_GROUP_ID"],
)

consume_loop(consumer, handle_job)