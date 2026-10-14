FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    librdkafka-dev \
    gcc \
    g++ \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://gitlab.eclipse.org/eclipse/aidge/aidge.git

RUN cd aidge \
    && ./setup.sh -m aidge_core -m aidge_backend_cpu -m aidge_onnx -m export_cpp

RUN cd aidge/aidge/aidge_quantization \
    && pip install --no-deps . --config-settings cmake.define.AIDGE_ENABLE_CUDA=OFF
    
COPY main.py quant.py kafka_consumer.py minio_client.py ./

CMD ["python", "main.py"]