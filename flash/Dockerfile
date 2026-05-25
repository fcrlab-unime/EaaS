FROM espressif/idf:release-v5.4

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# Installa nel Python usato da ESP-IDF
RUN . /opt/esp/idf/export.sh && \
    python -m pip install --no-cache-dir --break-system-packages \
    confluent-kafka minio pyserial esptool

WORKDIR /app

COPY main.py kafka_consumer.py minio_client.py ./

CMD ["python", "main.py"]