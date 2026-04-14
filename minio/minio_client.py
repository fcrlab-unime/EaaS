import os
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

class MinioManager:
    def __init__(self):
        # Configurazione tramite variabili d'ambiente
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.user = os.getenv("MINIO_ROOT_USER", "minioadmin")
        self.password = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
        self.secure = os.getenv("MINIO_SECURE", "False").lower() == "true"
        
        # Inizializzazione Client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

    def upload_file(self, bucket_name, object_name, file_path):
        """Carica un file nel bucket specificato"""
        try:
            # Crea il bucket se non esiste
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' creato.")

            self.client.fput_object(bucket_name, object_name, file_path)
            print(f"File '{file_path}' caricato correttamente come '{object_name}' nel bucket '{bucket_name}'.")
        except S3Error as exc:
            print(f"Errore durante l'upload: {exc}")

    def download_file(self, bucket_name, object_name, destination_path):
        """Scarica un oggetto dal bucket"""
        try:
            self.client.fget_object(bucket_name, object_name, destination_path)
            print(f"Oggetto '{object_name}' scaricato in '{destination_path}'.")
        except S3Error as exc:
            print(f"Errore durante il download: {exc}")


if __name__ == "__main__":
    manager = MinioManager()
    # manager.upload_file("test-bucket", "mio-doc.pdf", "./documento.pdf")
    # manager.download_file("test-bucket", "mio-doc.pdf", "./scaricato.pdf")