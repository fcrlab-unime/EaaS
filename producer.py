#!KAFKA_BOOTSTRAP=localhost:9092 KAFKA_TOPIC_IN=train-jobs python producer_test.py
import json
import os
from confluent_kafka import Producer

producer = Producer({
    "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"),
})

job = {
    "job_id": "train_test_001",
    "model_script": "resnet_custom.py",
    "class_name": "MyResNet",
    "weights": "weights_v1.pth",
    "dataset": "mnist_data.zip",
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