#!KAFKA_BOOTSTRAP=localhost:9092 KAFKA_TOPIC_IN=train-jobs python producer.py
import json
import os
from confluent_kafka import Producer

producer = Producer({
    "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "redpanda:29092"),
})

job = {
    "job_id": "train_test_001",
    "model_script": "CCN1D.py",
    "class_name": "CCN1D",
    "model_params": {
        "input_channels": 122,
        "hidden_channels": 512,
        "num_layers": 4
    },
    "weights": "best_model.pth",
    "dataset": "train_scaled.csv",
    "epochs": 2
}

def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}] offset {msg.offset()}")

producer.produce(
    topic=os.environ.get("KAFKA_TOPIC_IN", "train-jobs"),
    value=json.dumps(job).encode("utf-8"),
    callback=delivery_report,
)

producer.flush()