## Reporte de Estado del Proyecto: Agente de IA "Raphael"

**Fecha del Reporte:** 25 de Mayo de 2025
**Versión del Reporte:** 1.0
**Autor(es):** Andrey Gonzalez (Vaskias) / Asistente IA
**Proyecto Principal:** Desarrollo del Agente de IA "Raphael"

### 1. Resumen 

Este reporte detalla el estado actual del proyecto "Raphael", un agente de Inteligencia Artificial personalizado diseñado para el análisis de imágenes de ecuaciones matemáticas y con planes de expansión para incluir módulos con funcionalidades adicionales. El proyecto se centra en una arquitectura modular, con un componente principal (Raphael-Core) y módulos especializados (como Rapheye para visión). Actualmente, Raphael-Core interactúa con los usuarios a través de un bot de Telegram utilizando webhooks, procesa solicitudes de análisis de imágenes invocando al módulo Rapheye (una Azure Function con un LLM multimodal), y gestiona la persistencia de datos en Azure Cosmos DB. El desarrollo ha superado desafíos significativos relacionados con el despliegue y la integración de componentes asíncronos. El objetivo inmediato es consolidar el despliegue actual y, a futuro, expandir Raphael-Core con nuevos módulos y funcionalidades.

### 2. Objetivos del Proyecto

**2.1. Objetivo General:**
Desarrollar un agente de IA robusto, modular y extensible llamado "Raphael" capaz de proporcionar asistencia inteligente en diversos dominios, comenzando con el análisis avanzado de ecuaciones matemáticas y expandiéndose a otras áreas de conocimiento y herramientas de productividad.

**2.2. Objetivos Específicos (Fase Actual):**
*   **Análisis de Ecuaciones:** Implementar la capacidad de recibir imágenes de ecuaciones matemáticas, extraer su contenido (LaTeX, variables, etc.) y proporcionar un análisis detallado utilizando un LLM multimodal.
*   **Interacción con el Usuario:** Proveer una interfaz de usuario intuitiva a través de un bot de Telegram.
*   **Arquitectura Modular:** Diseñar el sistema con módulos claramente definidos para facilitar el mantenimiento, la escalabilidad y la adición de nuevas funcionalidades.
    *   **Raphael-Core:** Módulo orquestador principal.
    *   **Rapheye:** Módulo especializado en visión y análisis de ecuaciones.
*   **Persistencia de Datos:** Almacenar y recuperar análisis de ecuaciones para optimizar el rendimiento y evitar procesamientos redundantes.
*   **Despliegue Estable:** Lograr un despliegue funcional y estable del módulo Raphael-Core en un entorno de nube (Azure).

**2.3. Objetivos a Futuro (Próximas Fases):**
*   **Expansión de Módulos:** Integrar nuevos módulos a Raphael-Core que provean funcionalidades adicionales (ej. resolución de problemas, generación de código, resúmenes de texto, consultas a bases de conocimiento, etc.).
*   **Mejora de la IA:** Incorporar técnicas de IA más avanzadas, incluyendo memoria a largo plazo, aprendizaje continuo (si aplica), y personalización avanzada.
*   **Diversificación de Interfaces:** Explorar otras interfaces de usuario además de Telegram (ej. aplicación web, extensiones de navegador).
*   **Optimización y Escalabilidad:** Mejorar el rendimiento, la eficiencia de costos y la capacidad de escalado de todos los componentes del sistema.
*   **Seguridad y Robustez:** Fortalecer la seguridad de las APIs, la gestión de datos y la resiliencia general del sistema.

### 3. Arquitectura y Componentes del Sistema

**3.1. Diagrama de Arquitectura General:**
El sistema sigue una arquitectura de microservicios/módulos. El usuario interactúa con el **Bot de Telegram**. Los mensajes del bot son recibidos por **Raphael-Core** a través de un endpoint de Webhook. Raphael-Core procesa la solicitud:
    *   Si es una imagen de una ecuación, invoca al módulo **Rapheye** (Azure Function) mediante una llamada API HTTP.
    *   Rapheye analiza la imagen usando un LLM multimodal y devuelve un JSON con el análisis.
    *   Raphael-Core puede consultar/almacenar este análisis en **Azure Cosmos DB**.
    *   Finalmente, Raphael-Core formatea la respuesta y la envía de vuelta al usuario vía Telegram.

**3.2. Módulo de Visión "Rapheye"**

    *   **Propósito:** Análisis de imágenes de ecuaciones.
    *   **Tecnología:** Azure Function (Python v2).
    *   **Motor IA:** LLM Multimodal (ej. Google Gemini Pro/Flash).
    *   **Entrada (API):** Petición POST HTTP con bytes de imagen (ej. `image/jpeg`).
    *   **Salida (API):** JSON estructurado en español con: `latex_extracted_from_image`, `name`, `category`, `description`, `derivation`, `uses`, `vars`, `similars`.
    *   **Estado Actual:** Funcional y desplegado en Azure Functions. Se dispone de una URL de invocación válida con su respectiva clave de función.
    *   **URL Ejemplo:** `https://<rapheye-app-name>.azurewebsites.net/api/analyze_equation_from_image?code=<FUNCTION_KEY>`

**3.3. Módulo Principal "Raphael-Core"**

    *   **Propósito:** Orquestador principal, interfaz con Telegram, lógica de negocio, persistencia.
    *   **Tecnología Primaria:** Python, FastAPI, Uvicorn.
    *   **Interacción con Telegram:** Biblioteca `python-telegram-bot` (versión >= 20.x), configurada en modo **Webhook**.
        *   Un endpoint HTTP en FastAPI (`/{TOKEN_ID}_{WEBHOOK_SECRET_PATH_SUFFIX}`) recibe los `Update`s de Telegram.
    *   **Comunicación con Rapheye:** Cliente HTTP asíncrono `httpx` para llamadas API.
    *   **Base de Datos:** Azure Cosmos DB (API MongoDB) a través de `pymongo`.
        *   Almacena análisis de ecuaciones para reutilización.
        *   Clave principal para búsqueda: LaTeX normalizado.
    *   **Estructura Interna Clave:**
        *   **`TaskContext` (Pydantic Model):** Encapsula todos los datos relevantes para el procesamiento de una solicitud de imagen (IDs de chat/usuario/mensaje, imagen, análisis de Rapheye, resultado de la DB, etc.).
        *   **`orchestrate_image_processing(TaskContext)`:** Función principal que coordina la lógica de análisis de imágenes: llamada a Rapheye, consulta/escritura en DB, preparación de la respuesta.
        *   **Handlers de Telegram:** Funciones asíncronas para `CommandHandler` (ej. `/start`, `/help`) y `MessageHandler` (ej. para `filters.PHOTO`, `filters.TEXT`).
        *   **`lifespan` de FastAPI:**
            *   **Startup:** Inicializa la conexión a CosmosDB, configura la instancia de `Application` de `python-telegram-bot` (crea el objeto, añade handlers), y registra el webhook con la API de Telegram (`setWebhook`).
            *   **Shutdown:** Elimina el webhook de Telegram (`deleteWebhook`), cierra la instancia de `Application` de PTB (`ptb_application.shutdown()`), y cierra la conexión a CosmosDB.
    *   **Estado Actual:** Funcional en entorno de desarrollo local (`python main.py` con Uvicorn y `ngrok` para webhooks). Despliegue en Azure App Service para Contenedores en progreso y funcionando con la versión 0.10.1.

**3.4. Base de Datos (Azure Cosmos DB)**

    *   **API:** MongoDB.
    *   **Colección Principal:** `equations` (o según variable de entorno).
    *   **Estructura del Documento (Ejemplo):** Basada en el modelo `EquationAnalysis` de Pydantic, incluyendo `equation_id` (como `_id`), `latex`, `name`, `category`, `description`, `derivation`, `uses`, `vars`, `similars`, `llm_analysis_status`, `database_status`, y `normalized_latex_key`.

### 4. Estado del Desarrollo y Despliegue

**4.1. Funcionalidad Implementada:**

*   Recepción de imágenes de ecuaciones vía Telegram.
*   Llamada exitosa a la Azure Function "Rapheye" para análisis de imagen.
*   Procesamiento de la respuesta JSON de Rapheye.
*   Normalización de LaTeX.
*   Interacción (lectura/escritura) con Azure Cosmos DB para persistir/recuperar análisis.
*   Renderizado de LaTeX a imagen (usando Matplotlib) para la respuesta.
*   Respuesta al usuario en Telegram con el análisis detallado y la imagen renderizada.
*   Manejo de comandos básicos (`/start`, `/help`).
*   Configuración de logging detallado.

**4.2. Entorno de Desarrollo Local:**
*   Funciona correctamente con `python main.py` (que inicia Uvicorn) y `ngrok` para exponer el endpoint del webhook.
*   Variables de entorno gestionadas mediante archivo `.env` y `python-dotenv`.
*   Entorno virtual (`venv`) para dependencias.

**4.3. Despliegue (Enfoque Actual: Azure App Service para Contenedores con Webhooks):**
*   **Contenerización (Docker):**
    *   `Dockerfile` configurado para usar `python:3.10-slim` como base.
    *   Instala dependencias del sistema (ej. para `fontconfig`, `libjpeg`) y de Python (`requirements.txt`).
    *   Copia el código de la aplicación.
    *   Expone el puerto `8000`.
    *   Utiliza `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]` para ejecutar la aplicación.
*   **Azure Container Registry (ACR):**
    *   Se ha creado una instancia de ACR (ej. `raphaelcorexrapheye`) para almacenar las imágenes Docker.
    *   La imagen de `raphael-core-webhook` (tag `v0.10.1` o similar) ha sido subida exitosamente.
*   **Azure App Service para Contenedores:**
    *   Se ha creado un Plan de App Service (Linux, SKU B1).
    *   Se ha creado una Web App configurada para usar la imagen desde ACR.
    *   Variables de entorno configuradas en App Service, incluyendo `WEBHOOK_URL_BASE` apuntando a la URL `*.azurewebsites.net` de la Web App.
    *   `WEBSITES_PORT` configurado a `8000`.
    *   **Estado Actual del Despliegue:** El webhook se registra exitosamente con Telegram. La aplicación recibe los updates y los handlers se activan. La funcionalidad principal está operativa en Azure.

**4.4. Desafíos Superados (Historial Relevante):**
*   **Polling y Conflictos de Event Loop/Token en Docker:** Se experimentaron dificultades persistentes al intentar desplegar `Raphael-Core` con `python-telegram-bot` en modo polling dentro de contenedores Docker (especialmente con Gunicorn, pero también con Uvicorn directo en algunos casos). Los problemas incluían:
    *   Errores de `RuntimeError: Cannot close a running event loop` o `this event loop is already running`.
    *   El `Dispatcher` de PTB no activaba los handlers a pesar de que el polling recibía los `Update`s.
*   **Decisión de Migrar a Webhooks:** Debido a los problemas con el polling en Docker, se tomó la decisión estratégica de migrar la comunicación con Telegram a **Webhooks**, lo cual ha resultado en un despliegue exitoso y funcional en Azure App Service.

### 5. Configuración y Uso

**5.1. Variables de Entorno Requeridas (para Raphael-Core):**
*   `TELEGRAM_BOT_TOKEN`: Token del bot de Telegram.
*   `WEBHOOK_URL_BASE`: URL HTTPS pública base donde la aplicación FastAPI está escuchando (ej. `https://<tu-app>.azurewebsites.net` o URL de `ngrok`).
*   `COSMOS_MONGO_CONNECTION_STRING`: Cadena de conexión para Azure Cosmos DB.
*   `DATABASE_NAME`: Nombre de la base de datos en Cosmos DB (ej. `raphaeldb`).
*   `COLLECTION_NAME`: Nombre de la colección para las ecuaciones (ej. `equations`).
*   `RAPHEYE_VISION_FUNCTION_URL`: URL completa (con clave) de la Azure Function Rapheye.
*   `PYTHONUNBUFFERED=1` (Recomendado en contenedores para logs).
*   `WEBSITES_PORT=8000` (Para Azure App Service, si el contenedor expone en el puerto 8000).
*   (Opcional) `DELETE_WEBHOOK_ON_SHUTDOWN="true"` o `"false"`.

**5.2. Instrucciones de Uso del Bot:**
*   Interactuar con el bot en Telegram.
*   Comandos: `/start`, `/help`.
*   Funcionalidad principal: Enviar una imagen de una ecuación para análisis.

### 6. Próximos Pasos y Hoja de Ruta (Roadmap)

**6.1. Inmediatos:**
*   **Refactorizar `handle_photo` para usar `BackgroundTasks` de FastAPI:** Para asegurar que el endpoint del webhook responda rápidamente a Telegram, incluso si el procesamiento de la imagen (`orchestrate_image_processing`) es largo.
*   **Pruebas Exhaustivas del Despliegue Actual:** Validar todas las funcionalidades en el entorno de Azure App Service.
*   **Optimización de Logs:** Ajustar niveles de logging para producción (ej. `INFO` para la mayoría, `DEBUG` solo para componentes específicos si es necesario).
*   **Documentación Final (Manual y README):** Completar y refinar.

**6.2. Corto Plazo:**
*   **Seguridad del Webhook:** Implementar la verificación de un `secret_token` (proporcionado por Telegram en `setWebhook` y enviado en el header `X-Telegram-Bot-Api-Secret-Token`) en el endpoint de Raphael-Core.
*   **Manejo de Errores Mejorado:** Implementar notificaciones más detalladas al usuario y potentially reintentos para fallos en llamadas a servicios externos.
*   **CI/CD Pipeline:** Configurar un pipeline (ej. GitHub Actions, Azure DevOps) para automatizar el build de la imagen Docker, la subida a ACR y el despliegue/actualización en Azure App Service.

**6.3. Medio Plazo (Expansión de Funcionalidades - Nuevos Módulos):**
*   **Definición de Nuevos Módulos:**
    *   Identificar funcionalidades específicas (ej. resolución de ecuaciones algebraicas, graficación de funciones, consulta de bases de conocimiento específicas, traducción matemática).
    *   Diseñar la API para cada nuevo módulo (podrían ser Azure Functions, otros microservicios, o incluso librerías integradas directamente si son ligeras).
*   **Integración en Raphael-Core:**
    *   Añadir nuevos handlers de Telegram (comandos o tipos de mensaje) para invocar estas nuevas funcionalidades.
    *   Modificar o extender `TaskContext` si es necesario para manejar los datos de los nuevos módulos.
    *   Actualizar la lógica de orquestación en Raphael-Core para llamar a estos nuevos módulos.
*   **Ejemplo de Módulo Potencial: "Raphael-Solver"**
    *   Propósito: Resolver ecuaciones simbólicamente.
    *   Tecnología: Podría usar SymPy directamente dentro de Raphael-Core o ser un microservicio separado si el cómputo es intensivo.
    *   Interacción: El usuario podría enviar una ecuación en LaTeX (obtenida de Rapheye o escrita) y pedir su solución.

**6.4. Largo Plazo:**
*   Investigar y prototipar nuevas interfaces de usuario.
*   Explorar técnicas de IA más avanzadas para el agente.
*   Optimización de costos y rendimiento a escala.

### 7. Riesgos y Desafíos Anticipados

*   **Gestión de la Complejidad:** A medida que se añaden más módulos, mantener la cohesión y la simplicidad de la arquitectura de Raphael-Core será un desafío.
*   **Costos de Azure:** El uso de múltiples servicios (App Service, ACR, Cosmos DB, Functions, LLMs) incurrirá en costos que deben ser monitorizados y optimizados.
*   **Seguridad:** Con más funcionalidades y potencialmente más datos sensibles, la seguridad de las APIs y la protección de datos se vuelven más críticas.
*   **Mantenimiento de Múltiples Modelos LLM:** Si diferentes módulos usan diferentes LLMs, gestionar sus APIs, prompts y versiones puede ser complejo.
*   **Latencia:** Algunas funcionalidades nuevas podrían introducir latencia adicional en la respuesta al usuario. Se necesitará optimización y posiblemente el uso extensivo de tareas en segundo plano.

### 8. Conclusión

El proyecto Raphael ha alcanzado un hito importante al lograr un despliegue funcional de su núcleo (Raphael-Core) con el módulo de visión (Rapheye) utilizando webhooks de Telegram y Azure App Service. Los desafíos iniciales con el polling en Docker han sido superados mediante la migración a webhooks. El sistema ahora está bien posicionado para la siguiente fase de desarrollo, que se centrará en la refactorización para producción, la mejora de la robustez y, fundamentalmente, la expansión de Raphael-Core con nuevos módulos y capacidades inteligentes. La claridad de la arquitectura actual y la documentación detallada serán clave para el éxito de estas futuras fases, especialmente al colaborar con asistentes de IA para el desarrollo.

