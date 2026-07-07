FROM espressif/idf:release-v5.4

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV IDF_PATH=/opt/esp/idf

RUN apt-get update && apt-get install -y --no-install-recommends \
    librdkafka-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/esp/python_env/idf5.4_py3.12_env/bin/pip install --no-cache-dir \
    confluent-kafka minio pyserial esptool

WORKDIR /app

COPY main.py kafka_consumer.py minio_client.py ./

CMD ["/bin/bash", "-c", ". /opt/esp/idf/export.sh && python3 main.py"]