import json
import time
import logging
import os
from confluent_kafka import Consumer, KafkaError, Producer

logger = logging.getLogger(__name__)


def make_consumer(bootstrap: str, topic: str, group_id: str) -> Consumer:
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    for attempt in range(10):
        try:
            consumer.subscribe([topic])
            consumer.poll(timeout=2.0)
            return consumer
        except Exception as e:
            logger.warning("Kafka not ready, retry %d/10: %s", attempt + 1, e)
            time.sleep(3)
    consumer.subscribe([topic])
    return consumer


def consume_loop(consumer: Consumer, handler):
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            print(f"[{time.strftime('%H:%M:%S')}] Messaggio ricevuto!")

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka error: %s", msg.error())
                continue

            try:
                job = json.loads(msg.value().decode("utf-8"))
                logger.info("Received job: %s", job.get("job_id"))
                handler(job)
                consumer.commit(message=msg)
            except Exception as e:
                logger.exception("Failed to process message: %s", e)

    finally:
        consumer.close()


def _delivery_report(err, msg):
    if err:
        logger.error("Delivery failed: %s", err)
    else:
        logger.info("Delivered to %s [%d] offset %d", msg.topic(), msg.partition(), msg.offset())


def send_kafka_message(topic: str, message: dict):
    producer = Producer({
        "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "redpanda:29092"),
    })
    producer.produce(
        topic=topic,
        value=json.dumps(message).encode("utf-8"),
        callback=_delivery_report,
    )
    producer.flush()
