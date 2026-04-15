import json
import logging
from confluent_kafka import Consumer, KafkaError

logger = logging.getLogger(__name__)

def make_consumer(bootstrap: str, topic: str, group_id: str) -> Consumer:
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([topic])
    return consumer


def consume_loop(consumer: Consumer, handler):
    """
    handler(job: dict) -> None
    Chiamato per ogni messaggio valido. Commit solo dopo handler completato.
    """
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
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
                # non committa: il messaggio verrà riprocessato

    finally:
        consumer.close()