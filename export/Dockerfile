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
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone --depth 1 https://gitlab.eclipse.org/eclipse/aidge/aidge.git \
    && cd aidge \
    && ./setup.sh -m aidge_core -m aidge_backend_cpu -m aidge_export_cpp

COPY . .

CMD ["python", "main.py"]
