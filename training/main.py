import os
import sys
import json
import importlib.util
import logging
import tarfile
import torch
import shutil
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
        logger.info(f"🚀 Inizio elaborazione Job: {job_id}")

        # --- 1. DOWNLOAD DA MINIO ---
        # Scarichiamo lo script dell'architettura
        script_path = download(minio, "models", job["model_script"], str(base_dir / "model_script.py"))
        
        # Scarichiamo il dataset (CSV)
        dataset_local = str(base_dir / job["dataset"])
        download(minio, "datasets", job["dataset"], dataset_local)

        # Scarichiamo i pesi solo se presenti nel messaggio
        weights_path = None
        if job.get("weights"):
            weights_path = download(minio, "weights", job["weights"], str(base_dir / "weights.pth"))

        # --- 2. ISTANZIAZIONE MODELLO ---
        # Carichiamo la classe dal file .py scaricato
        ModelClass = load_model_class(str(script_path), class_name)
        
        # Istanziamo passando i parametri estratti dal JSON di Kafka
        # Esempio: ModelClass(input_channels=1, num_layers=3)
        model = ModelClass(**model_params)
        logger.info(f"✅ Modello {class_name} istanziato con successo.")

        # --- 3. CARICAMENTO PESI INIZIALI ---
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
            logger.info(f"📥 Pesi iniziali caricati da {job['weights']}")

        # --- 4. TRAINING ---
        # Passiamo il modello e il percorso del CSV alla funzione di train
        model = train(model, dataset_local, epochs=job["epochs"])

        # --- 5. EXPORT ONNX ---
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
        logger.info(f"✅ ONNX esportato: {onnx_path}")

        # --- 6. SALVATAGGIO E METADATI ---
        output_path = base_dir / "trained_model.pth"
        torch.save({...}, str(output_path))

        # --- 7. UPLOAD RISULTATI ---
        minio_pth_dest  = f"trained/{job_id}/{class_name}_final.pth"
        minio_onnx_dest = f"trained/{job_id}/{class_name}_final.onnx"

        upload(minio, "models", minio_pth_dest,  str(output_path))
        upload(minio, "models", minio_onnx_dest, str(onnx_path))

        logger.info(f"📤 .pth caricato in: {minio_pth_dest}")
        logger.info(f"📤 .onnx caricato in: {minio_onnx_dest}")

    except Exception as e:
        logger.error(f"❌ Errore critico nel Job {job_id}: {str(e)}", exc_info=True)
        # Qui potresti inviare un messaggio Kafka di errore su un topic di 'status'
    
    finally:
        # --- 7. CLEANUP FINALE ---
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.info(f"🧹 Pulizia cartella temporanea completata per {job_id}")
            

      
consumer = make_consumer(
    bootstrap=os.environ["KAFKA_BOOTSTRAP"],
    topic=os.environ["KAFKA_TOPIC_IN"],
    group_id=os.environ["KAFKA_GROUP_ID"],
)

consume_loop(consumer, handle_job)