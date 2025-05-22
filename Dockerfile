# Usar una imagen base oficial de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar solo el archivo de requisitos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Actualizar pip e instalar las dependencias de Python
# Si Matplotlib o Pillow fallan aquí, podríamos necesitar añadir algunas dependencias de sistema
# con apt-get ANTES de este paso (ej. libfreetype6-dev, libpng-dev, g++)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación al directorio de trabajo
# (Asegúrate que tu .dockerignore excluye venv, __pycache__, .git, .env, etc.)
COPY . .

# Exponer el puerto en el que Gunicorn/Uvicorn escucharán
EXPOSE 8000

# Comando para ejecutar la aplicación cuando el contenedor se inicie
# Usando el main.py (v0.6.5 que me diste) que tiene el lifespan llamando a run_telegram_polling
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--log-level", "info"]