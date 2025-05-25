# Usar una imagen base oficial de Python. python:3.10-slim es una buena elección.
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Establecer variables de entorno para Python (buenas prácticas)
ENV PYTHONDONTWRITEBYTECODE 1  # Evita que Python escriba archivos .pyc (útil en contenedores)
ENV PYTHONUNBUFFERED 1       # Asegura que la salida de Python (print, logs) se muestre inmediatamente

# Copia el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instala las dependencias del sistema que puedan ser necesarias
# Estas son para Matplotlib y Pillow, principalmente.
# Si no usas Matplotlib/Pillow directamente en este servicio, podrías omitirlas
# o reducirlas, pero no hacen daño si están.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    # tk-dev libส่วนตัว # Esto parecía ser un error de pegado, no creo que lo necesites
    && rm -rf /var/lib/apt/lists/*
    
# Actualiza pip e instala las dependencias de Python desde requirements.txt
# --no-cache-dir ayuda a reducir el tamaño de la imagen
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de tu aplicación al directorio de trabajo
# Asegúrate de tener un .dockerignore para excluir archivos innecesarios (venv, __pycache__, .git, etc.)
COPY . .

# Expone el puerto en el que Uvicorn escuchará
# Este es el puerto DENTRO del contenedor.
EXPOSE 8000

# Comando para ejecutar la aplicación cuando el contenedor se inicie
# Se usa Uvicorn directamente, lo cual es bueno para FastAPI y maneja bien asyncio.
# --host 0.0.0.0 : Hace que Uvicorn escuche en todas las interfaces de red dentro del contenedor,
#                   necesario para que sea accesible desde fuera del contenedor.
# --port 8000    : El puerto en el que Uvicorn escuchará dentro del contenedor (coincide con EXPOSE).
# --workers 1    : Importante. Aunque Uvicorn puede manejar múltiples workers, para un bot de Telegram
#                  (incluso con webhooks, donde el estado de la `Application` de PTB es importante)
#                  y para simplificar, empezar con 1 worker es más seguro y predecible.
#                  Si la carga de la API FastAPI se vuelve muy alta, se pueden explorar arquitecturas
#                  más complejas, pero para el bot, 1 worker es ideal.
# --log-level debug: Muy útil para depuración. Puedes cambiarlo a "info" o "warning" en producción.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "debug"]