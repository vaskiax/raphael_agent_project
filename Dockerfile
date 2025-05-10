# Usar una imagen base de Python 3.10 slim
FROM python:3.10-slim

# Instalar dependencias sistema: git Y LAS NECESARIAS PARA MATPLOTLIB
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Instalar dependencias Python (requirements.txt ahora incluye matplotlib)
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copiar código
COPY main.py .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]