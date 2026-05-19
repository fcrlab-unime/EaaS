FROM python:3.12-slim

# Impedisce a Python di generare file .pyc e forza il flush dei log
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Installazione dipendenze di sistema (Aggiunto git e cmake spesso richiesto da Aidge)
RUN apt-get update && apt-get install -y \
    librdkafka-dev \
    gcc \
    g++ \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 3. Installazione dipendenze dal tuo file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Clonazione e build di Aidge
# NOTA: Usiamo --depth 1 per clonare solo l'ultimo commit e risparmiare spazio
RUN git clone --depth 1 https://gitlab.eclipse.org/eclipse/aidge/aidge.git \
    && cd aidge \
    && ./setup.sh -m aidge_core -m aidge_backend_cpu -m aidge_onnx -m aidge_quantization -m export_cpp
    

# 5. Copia del codice sorgente
COPY . .

CMD ["python", "main.py"]