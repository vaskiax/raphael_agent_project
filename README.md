# Agente de IA "Raphael" - Asistente Matemático

Raphael es un agente de Inteligencia Artificial personalizado diseñado para el análisis de imágenes de ecuaciones matemáticas. Proporciona una descomposición detallada, incluyendo la extracción de LaTeX, nombre, categoría, descripción, y más, utilizando un LLM Multimodal.

**Versión Actual del Core:** 0.10.x (Modo Webhook)

## Arquitectura General

El sistema Raphael consta de dos módulos principales:

1.  **Rapheye (Módulo de Visión):**
    *   Una Azure Function (Python) que utiliza un LLM Multimodal (ej. Google Gemini) para realizar OCR y análisis semántico de imágenes de ecuaciones.
    *   Entrada: Bytes de imagen.
    *   Salida: JSON estructurado con el análisis de la ecuación.
2.  **Raphael-Core (Módulo Principal y Bot):**
    *   Una aplicación FastAPI (Python) que actúa como orquestador.
    *   Interactúa con los usuarios a través de un bot de Telegram (configurado con **Webhooks**).
    *   Llama a Rapheye para el análisis de imágenes.
    *   Persiste y consulta análisis en Azure Cosmos DB (API MongoDB).
    *   Renderiza LaTeX a imágenes usando Matplotlib.

![Diagrama de Arquitectura (Placeholder)](./path_a_tu_diagrama_si_lo_tienes.png)
*(Reemplaza la línea anterior con un diagrama de tu arquitectura si lo tienes)*

## Tecnologías Utilizadas

*   **Backend:** Python, FastAPI, Uvicorn
*   **Bot:** `python-telegram-bot` (modo Webhook)
*   **Visión IA:** Azure Functions, LLM Multimodal (ej. Gemini)
*   **Base de Datos:** Azure Cosmos DB (API MongoDB), Pymongo
*   **Otros:** Pillow, Matplotlib, SymPy (opcional), `python-dotenv`
*   **Despliegue (ejemplo):** Docker, Azure App Service para Contenedores

## Configuración del Entorno de Desarrollo (Raphael-Core)

### Prerrequisitos
*   Python 3.10+
*   Git
*   `pip` y `venv`
*   `ngrok` (para pruebas de webhook local)

### Pasos
1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_CARPETA_RAPHAEL_CORE>
    ```
2.  **Crear y activar entorno virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows (cmd)
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
4.  **Configurar variables de entorno:**
    Crea un archivo `.env` en la raíz del proyecto Raphael-Core y llénalo con tus credenciales y URLs (ver `main.py` o el manual para los nombres de las variables requeridas):
    ```env
    TELEGRAM_BOT_TOKEN="TU_TOKEN_DE_TELEGRAM"
    WEBHOOK_URL_BASE="TU_URL_HTTPS_DE_NGROK_O_PRODUCCION" # SIN barra al final
    COSMOS_MONGO_CONNECTION_STRING="TU_CADENA_DE_COSMOS_DB"
    DATABASE_NAME="raphaeldb"
    COLLECTION_NAME="equations"
    RAPHEYE_VISION_FUNCTION_URL="URL_DE_TU_AZURE_FUNCTION_RAPHEYE"
    # Opcional: DELETE_WEBHOOK_ON_SHUTDOWN="true"
    ```
5.  **Ejecutar localmente (para desarrollo con webhooks):**
    *   **Terminal 1 (ngrok):**
        ```bash
        ngrok http 8000 
        ```
        (Asegúrate de que la URL HTTPS de ngrok esté en `WEBHOOK_URL_BASE` en tu `.env`)
    *   **Terminal 2 (Aplicación FastAPI/Uvicorn):**
        ```bash
        python main.py
        # o
        # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ```
    La aplicación intentará registrar el webhook con Telegram al iniciar.

## Despliegue (Ejemplo con Azure App Service para Contenedores)

1.  **Azure Container Registry (ACR):**
    *   Crea una instancia de ACR (puedes usar el script `raphael_setup.sh` o el Portal/Azure CLI).
    *   Inicia sesión: `az acr login --name <TU_ACR_NAME>`
2.  **Docker:**
    *   Asegúrate de que tu `Dockerfile` y `.dockerignore` estén configurados.
    *   Construye la imagen: `docker build -t <LOGIN_SERVER_ACR>/raphael-core-webhook:<TAG> .`
    *   Sube la imagen: `docker push <LOGIN_SERVER_ACR>/raphael-core-webhook:<TAG>`
3.  **Azure App Service para Contenedores:**
    *   Crea un Plan de App Service (Linux, SKU B1 o similar).
    *   Crea una Web App para Contenedores usando la imagen de tu ACR.
    *   Configura las variables de entorno en App Service, incluyendo `WEBHOOK_URL_BASE` con la URL `https://<tu-app-name>.azurewebsites.net`.
    *   Asegúrate de que `WEBSITES_PORT` esté configurado a `8000` (o el puerto que Uvicorn expone).
    *   Revisa los logs del "Flujo de registro" en App Service.
    *   (Consulta el script `deploy_raphael_app_service.sh` para ver un ejemplo de automatización con Azure CLI).

## Uso del Bot de Telegram

*   **`/start`**: Inicia la conversación.
*   **`/help`**: Muestra ayuda.
*   **Enviar una foto de una ecuación**: El bot la analizará y responderá con detalles y una imagen renderizada del LaTeX.

## bot ejemplo

@low_raphael_bot

## Licencia

vaskias