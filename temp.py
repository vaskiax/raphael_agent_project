# -*- coding: utf-8 -*-
# Versión 0.6.0 - Integración con Azure Function para análisis de imágenes
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File
import asyncio
# requests ya no es necesario para la llamada a Gemini, httpx lo reemplaza para la Azure Function
import os
import uuid
from pydantic import BaseModel, ValidationError, Field # Field ya lo tenías indirectamente por los modelos
import logging
import json
import re
import io
from PIL import Image
import html

# --- Matplotlib para renderizar LaTeX ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Database Imports ---
import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure, InvalidName

# --- HTTPX para llamar a la Azure Function ---
import httpx # <--- AÑADIDO

# --- Google Gemini Imports (YA NO SON NECESARIOS AQUÍ) ---
# import google.generativeai as genai               # <--- ELIMINADO
# from google.generativeai.types import GenerationConfig # <--- ELIMINADO
from dotenv import load_dotenv

# --- SymPy Import ---
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy import latex as sympy_latex
    sympy_available = True
except ImportError:
    sympy_available = False
    logging.warning("SymPy no encontrado.")
    parse_latex = None
    sympy_latex = None

# --- Telegram Bot Imports ---
import telegram
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, TypeHandler, filters
)
from telegram.constants import ParseMode

# --- Configuración ---
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING) # httpx logger ya estaba
logging.getLogger("matplotlib").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- Crear Instancia FastAPI ---
app = FastAPI(title="Raphael Agent API & Telegram Bot", version="0.6.0") # Versión actualizada

# --- Variables de Entorno ---
# GOOGLE_API_KEY ya no es necesaria aquí si toda la lógica de Gemini está en la Azure Function
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # <--- COMENTADO/ELIMINADO
MONGO_CONNECTION_STRING = os.getenv("COSMOS_MONGO_CONNECTION_STRING")
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "raphaeldb")
COLLECTION_NAME = os.getenv("COSMOS_COLLECTION_NAME", "equations")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAPHEYE_VISION_FUNCTION_URL = os.getenv("RAPHEYE_VISION_FUNCTION_URL") # <--- AÑADIDO

# logger.info(f"GOOGLE_API_KEY loaded: {'Yes' if GOOGLE_API_KEY else 'NO'}") # <--- COMENTADO/ELIMINADO
logger.info(f"COSMOS_MONGO_CONNECTION_STRING loaded: {'Yes' if MONGO_CONNECTION_STRING else 'NO'}")
logger.info(f"TELEGRAM_BOT_TOKEN loaded: {'Yes' if TELEGRAM_BOT_TOKEN else 'NO'}")
logger.info(f"RAPHEYE_VISION_FUNCTION_URL loaded: {'Yes' if RAPHEYE_VISION_FUNCTION_URL else 'NO'}") # <--- AÑADIDO

if not RAPHEYE_VISION_FUNCTION_URL:
    logger.critical("¡¡¡ RAPHEYE_VISION_FUNCTION_URL no está configurada!!! El análisis de imágenes no funcionará.")


# --- Configurar Google Gemini (YA NO ES NECESARIO AQUÍ) ---
# genai_configured = False
# gemini_model_multimodal = None
# if GOOGLE_API_KEY:
#     try:
#         genai.configure(api_key=GOOGLE_API_KEY)
#         gemini_model_multimodal = genai.GenerativeModel('gemini-2.0-flash')
#         logger.info("Google AI SDK configurado y modelo 'gemini-2.0-flash-latest' cargado.") # Era gemini-2.0-flash
#         genai_configured = True
#     except Exception as e:
#         logger.error(f"Error config Google AI SDK: {e}", exc_info=True)
# else:
#     logger.warning("GOOGLE_API_KEY no configurada para Gemini.")
# <--- TODA ESTA SECCIÓN ELIMINADA ---

# --- Configuración Base de Datos (Permanece igual) ---
db_client: pymongo.MongoClient | None = None
db_collection: pymongo.collection.Collection | None = None
if MONGO_CONNECTION_STRING:
    try:
        db_client = pymongo.MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=30000)
        db_client.admin.command('ping')
        if not DATABASE_NAME or any(c in DATABASE_NAME for c in [' ','.','$','/','\\','\0']):
             raise InvalidName(f"Nombre DB inválido: '{DATABASE_NAME}'")
        if not COLLECTION_NAME or '$' in COLLECTION_NAME or '\0' in COLLECTION_NAME:
             raise InvalidName(f"Nombre Colección inválido: '{COLLECTION_NAME}'")
        db = db_client[DATABASE_NAME]
        db_collection = db[COLLECTION_NAME]
        logger.info(f"Conectado a MongoDB/CosmosDB. DB: '{DATABASE_NAME}', Colección: '{COLLECTION_NAME}'")
    except Exception as e:
        logger.error(f"Error configuración/conexión MongoDB/CosmosDB: {e}", exc_info=True)
        db_client = None; db_collection = None
else:
    logger.warning("COSMOS_MONGO_CONNECTION_STRING no configurada.")

# --- Lista Permitida de Categorías (Permanece igual) ---
ALLOWED_CATEGORIES = [
    "mecánica clasica", "mecánica cuantica", "algebra lineal", "cálculo",
    "ecuaciones diferenciales parciales", "ecuaciones diferenciales ordinarias",
    "estadística y probabilidad", "termodinámica", "relatividad",
    "series (taylor, fourier, laurent, etc...)", "genericas", "óptica", "electromagnetismo"
]

# --- Modelos de Datos (Pydantic) ---
# LLMAnalysisData se usaba para el output directo de Gemini.
# Ahora recibiremos un JSON de la Azure Function. Si la Azure Function usa LLMResponseData
# y esa estructura es la que quieres usar aquí, puedes renombrar o ajustar.
# Por ahora, mantendré LLMAnalysisData y asumiré que la Azure Function devuelve
# campos con esos nombres.
class LLMAnalysisData(BaseModel): # Este modelo representa lo que ESPERAMOS de la Azure Function
    latex_extracted_from_image: str | None = None
    name: str | None = None
    category: str | None = None
    description: str | None = None
    derivation: str | None = None
    uses: str | None = None
    vars: str | None = None # Asumimos que la Azure Function devuelve esto como string
    similars: str | None = None # Asumimos que la Azure Function devuelve esto como string

class EquationAnalysis(BaseModel): # Este es tu modelo para la DB y la respuesta final, permanece.
    equation_id: str
    latex: str # Este debería ser el 'latex_extracted_from_image'
    name: str | None = None
    category: str | None = None
    description: str | None = None
    derivation: str | None = None
    uses: str | None = None
    vars: str | None = None
    similars: str | None = None
    llm_analysis_status: str
    database_status: str
    normalized_latex_key: str | None = None # Campo añadido para búsqueda


# --- Función de Normalización de LaTeX (Permanece igual) ---
def normalize_latex(latex_str: str) -> str | None:
    if not latex_str: return None
    normalized = re.sub(r'\s+', ' ', latex_str).strip()
    if normalized.startswith("$$") and normalized.endswith("$$"): normalized = normalized[2:-2].strip()
    if normalized.startswith("\\[") and normalized.endswith("\\]"): normalized = normalized[2:-2].strip()
    if normalized.startswith("$") and normalized.endswith("$"): normalized = normalized[1:-1].strip()
    if sympy_available and parse_latex and sympy_latex:
        try:
            sympy_expr = parse_latex(normalized)
            normalized_sympy = sympy_latex(sympy_expr, mode='inline', mul_symbol='dot') # o mode='plain'
            return normalized_sympy.strip()
        except Exception: return normalized # Devuelve pre-sympy si falla
    return normalized

# --- Función de Búsqueda en DB (Permanece igual) ---
async def find_equation_by_normalized_latex(normalized_latex_key: str) -> dict | None:
    if db_collection is None or not normalized_latex_key: return None
    try:
        return db_collection.find_one({"normalized_latex_key": normalized_latex_key})
    except Exception as e: logger.error(f"Error buscando en DB: {e}", exc_info=True); return None


# --- FUNCIÓN PARA LLAMAR A LA AZURE FUNCTION (NUEVA) ---
async def call_rapheye_vision_azure_function(image_bytes: bytes, image_mime_type: str) -> LLMAnalysisData | None:
    if not RAPHEYE_VISION_FUNCTION_URL:
        logger.error("RAPHEYE_VISION_FUNCTION_URL no está configurada. No se puede llamar a la función de análisis.")
        return None
    if not image_bytes:
        logger.error("Bytes de imagen vacíos para enviar a la Azure Function.")
        return None

    headers = {"Content-Type": image_mime_type}
    # Timeout generoso, la Azure Function tiene su propio timeout con Gemini.
    # Este timeout es para la comunicación con la Azure Function misma.
    timeout_config = httpx.Timeout(190.0, connect=15.0) # connect_timeout de 15s, total 190s

    try:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            logger.info(f"Enviando imagen a Azure Function: {RAPHEYE_VISION_FUNCTION_URL[:80]}...") # Loguea solo una parte
            response = await client.post(RAPHEYE_VISION_FUNCTION_URL, content=image_bytes, headers=headers)

        response.raise_for_status() # Lanza excepción para errores HTTP 4xx/5xx
        
        response_json = response.json()
        logger.info("Respuesta JSON recibida de Azure Function.")
        
        # Validar con Pydantic el JSON recibido de la Azure Function
        try:
            validated_data = LLMAnalysisData(**response_json)
            return validated_data
        except ValidationError as e:
            logger.error(f"Error de validación Pydantic para la respuesta de Azure Function: {e}")
            logger.debug(f"JSON de Azure Function que falló validación: {response_json}")
            return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Error HTTP {e.response.status_code} al llamar a Azure Function: {e.response.text[:500]}")
        return None
    except httpx.RequestError as e: # Errores de red, DNS, timeout de httpx, etc.
        logger.error(f"Error de red/solicitud al llamar a Azure Function: {type(e).__name__} - {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON de Azure Function: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
             logger.debug(f"Respuesta cruda de Azure Function que causó error JSON: {response.text[:500]}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado llamando a Azure Function: {e}", exc_info=True)
        return None

# --- Función para Llamar a Gemini CON IMAGEN (OCR + Análisis) ---
# <--- ESTA FUNCIÓN ENTERA (extract_and_analyze_equation_from_image_with_gemini) SE ELIMINA --- >
# async def extract_and_analyze_equation_from_image_with_gemini(...) -> LLMAnalysisData | None:
#   ... (todo el código antiguo aquí)


# --- Función para Guardar en MongoDB/Cosmos (Permanece igual) ---
async def save_analysis_to_mongo(analysis_data: EquationAnalysis, normalized_latex_key: str) -> bool:
    if db_collection is None: logger.warning("DB no configurada."); return False
    try:
        # Usar model_dump() para Pydantic v2 si es posible, o dict() para v1
        doc_to_save = analysis_data.model_dump() if hasattr(analysis_data, 'model_dump') else analysis_data.dict()
        doc_to_save['_id'] = analysis_data.equation_id # Asegurar _id
        doc_to_save['normalized_latex_key'] = normalized_latex_key # Asegurar normalized_latex_key

        res = db_collection.replace_one({'_id': analysis_data.equation_id}, doc_to_save, upsert=True)
        return res.acknowledged
    except Exception as e: logger.error(f"Error guardando DB: {e}", exc_info=True); return False

# --- Función para Renderizar LaTeX a Imagen (Permanece igual) ---
def render_latex_to_image_bytes(latex_str: str | None, filename: str = "equation.png") -> io.BytesIO | None:
    if not latex_str or not latex_str.strip():
        logger.warning(f"Render LaTeX inválido/vacío: {filename}")
        return None
    txt = latex_str.strip()
    if not (txt.startswith('$') and txt.endswith('$')) and not txt.startswith('\\['):
        txt = f"${txt}$" # Asegurar delimitadores si no están
    fig = None
    try:
        # Ajustar tamaño de figura dinámicamente puede ser complejo.
        # Un enfoque más simple pero robusto es usar un tamaño fijo y permitir 'wrap'.
        fig_height = max(2.5, len(txt.splitlines()) * 0.6 + 1.0) # Ajuste original
        fig_width = 10

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200) # dpi puede aumentarse para más calidad
        fig.patch.set_facecolor('white')
        ax.text(0.5, 0.5, txt, fontsize=18, ha='center', va='center', wrap=True)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
        buf.seek(0)
        logger.info(f"Renderizado LaTeX a PNG ok: {filename}")
        return buf
    except Exception as e:
        logger.error(f"Error renderizando LaTeX '{txt[:30]}...': {e}", exc_info=True)
        return None
    finally:
        if fig: plt.close(fig)


# --- Lógica Principal de Procesamiento (MODIFICADA) ---
async def process_equation(image_bytes: bytes, image_mime_type: str, filename: str = "telegram_image") -> EquationAnalysis | None:
    # 1. Llamar a la Azure Function para el análisis
    llm_data = await call_rapheye_vision_azure_function(image_bytes, image_mime_type) # <--- MODIFICADO

    if not llm_data or not llm_data.latex_extracted_from_image:
        logger.error("Fallo al obtener análisis de Azure Function o no se extrajo LaTeX.")
        # Opcional: devolver un objeto EquationAnalysis con estado de error para Telegram
        return EquationAnalysis(
            equation_id=str(uuid.uuid4()),
            latex="", # Opcional, o None. Depende de cómo manejes el None en Telegram.
            llm_analysis_status="Failed (Azure Function error or no LaTeX)",
            database_status="N/A - Analysis Failed"
            # ...otros campos None o valores por defecto
        )

    extracted_latex = llm_data.latex_extracted_from_image
    # Utiliza el normalized_latex_key para la búsqueda en DB
    # Si no lo genera la Azure Function (que no lo hace según su prompt), lo generamos aquí.
    normalized_key = normalize_latex(extracted_latex) or extracted_latex

    # 2. Buscar en la base de datos
    if db_collection is not None and (existing_doc_dict := await find_equation_by_normalized_latex(normalized_key)):
        try:
            # Necesitamos asegurar que los campos de estado se manejen bien al cargar desde DB.
            # Y que el 'latex' en EquationAnalysis sea el `latex_extracted_from_image`.
            data_for_model = {**existing_doc_dict} # Copia del diccionario
            data_for_model['latex'] = data_for_model.get('latex_extracted_from_image', data_for_model.get('latex')) # Prioriza el campo de la DB si existe, o el latex genérico
            data_for_model['database_status'] = "Retrieved from DB"
            data_for_model['llm_analysis_status'] = data_for_model.get('llm_analysis_status', 'Retrieved (Original status unknown)')
            
            # Para Pydantic, es mejor pasar solo los campos que el modelo espera.
            # Si `existing_doc_dict` tiene campos extra como `_id`, `normalized_latex_key`
            # que no están en `EquationAnalysis` (excepto equation_id), Pydantic podría quejarse
            # si `model_config` no tiene `extra='ignore'`.
            # Es más seguro construir el dict explícitamente o asegurar que EquationAnalysis los maneje.

            # Si `equation_id` no está y `_id` sí:
            if 'equation_id' not in data_for_model and '_id' in data_for_model:
                data_for_model['equation_id'] = str(data_for_model['_id'])

            # Eliminar _id si no es parte del modelo EquationAnalysis
            if '_id' in data_for_model and '_id' not in EquationAnalysis.model_fields:
                del data_for_model['_id']
            if 'normalized_latex_key' in data_for_model and 'normalized_latex_key' not in EquationAnalysis.model_fields:
                 del data_for_model['normalized_latex_key']


            logger.info(f"Ecuación encontrada en DB con normalized_key: {normalized_key}")
            return EquationAnalysis(**data_for_model)
        except ValidationError as e:
            logger.error(f"Error de validación Pydantic al cargar desde DB: {e}. Datos: {existing_doc_dict}. Se procederá a analizar de nuevo.")
            # Si falla la validación, es como si no lo hubiéramos encontrado, así que continuamos para analizar.
        except Exception as e:
            logger.error(f"Error inesperado al procesar documento de DB: {e}. Se procederá a analizar de nuevo.", exc_info=True)


    # 3. Si no está en DB o la validación falló, construir el objeto de análisis
    # Usar los datos de `llm_data` (que es `LLMAnalysisData` validado)
    analysis = EquationAnalysis(
        equation_id=str(uuid.uuid4()),
        latex=extracted_latex, # Campo principal para la ecuación
        name=llm_data.name,
        category=llm_data.category,
        description=llm_data.description,
        derivation=llm_data.derivation,
        uses=llm_data.uses,
        vars=llm_data.vars, # Este era un string único
        similars=llm_data.similars, # Este era un string único
        llm_analysis_status="Success (via Azure Function)", # Actualizado
        database_status="Save Attempt Pending",
        normalized_latex_key=normalized_key # Guardamos la clave usada para buscar/guardar
    )

    # 4. Guardar en la base de datos
    if db_collection is not None:
        if await save_analysis_to_mongo(analysis, normalized_key): # Pasamos la clave normalizada
            analysis.database_status = "Saved Successfully (New Entry)"
            logger.info(f"Análisis guardado en DB para normalized_key: {normalized_key}")
        else:
            analysis.database_status = "Save Failed"
            logger.warning(f"Fallo al guardar análisis en DB para normalized_key: {normalized_key}")
    elif db_collection is None:
        analysis.database_status = "Skipped (DB Unavailable)"

    return analysis


# --- Manejadores Telegram (Sin cambios funcionales mayores, pero dependen de process_equation) ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(f"¡Hola {update.effective_user.mention_html()}! Envíame una foto de una ecuación para analizarla.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html("Envíame una foto de una ecuación matemática. <b>Consejos:</b> Asegúrate de que la imagen sea clara y la ecuación esté bien enfocada.")
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Por favor, envíame una FOTO de una ecuación para que pueda ayudarte.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        logger.warning("Mensaje sin foto recibido en handle_photo.")
        return

    chat_id = update.effective_chat.id
    # Enviar mensaje de "procesando" y guardarlo para editarlo después
    processing_message = await context.bot.send_message(chat_id=chat_id, text="📸 Analizando la ecuación en la imagen, por favor espera...")

    try:
        # Obtener el archivo de la foto de mayor resolución
        photo_file = await update.message.photo[-1].get_file()
        
        # Descargar la imagen a un buffer en memoria
        image_bytes_io = io.BytesIO()
        await photo_file.download_to_memory(image_bytes_io)
        image_bytes = image_bytes_io.getvalue() # Bytes para enviar a la Azure Function

        # Determinar el MIME type de la imagen (Pillow)
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_format = pil_image.format or "JPEG" # Default a JPEG si no se puede determinar
            # Mapeo simple para MIME types comunes
            mime_type_map = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "BMP": "image/bmp"}
            image_mime_type = mime_type_map.get(image_format.upper(), "image/jpeg") # Default a jpeg
            logger.info(f"Imagen recibida. Formato detectado: {image_format}, MIME type: {image_mime_type}")
        except Exception as e_pil:
            logger.error(f"Error al procesar imagen con Pillow: {e_pil}. Usando image/jpeg por defecto.")
            image_mime_type = "image/jpeg" # Fallback seguro

        # Llamar a la función de procesamiento principal
        analysis_result = await process_equation(
            image_bytes,
            image_mime_type,
            filename=f"tg_{photo_file.file_id}.{image_format.lower()}"
        )

        if analysis_result and analysis_result.latex: # Si hay resultado Y se extrajo LaTeX
            message_parts = ["✅ <b>¡Análisis Completado!</b>"]
            if analysis_result.database_status == "Retrieved from DB":
                message_parts.append("<i>(Información recuperada de la base de datos)</i>")

            def add_field_to_message(label, value, is_code=False):
                if value and str(value).strip():
                    escaped_value = html.escape(str(value).strip())
                    if is_code:
                        message_parts.append(f"\n<b>{html.escape(label)}:</b>\n<code>{escaped_value}</code>")
                    else:
                        message_parts.append(f"\n<b>{html.escape(label)}:</b> {escaped_value}")
            
            add_field_to_message("Nombre", analysis_result.name)
            add_field_to_message("Categoría(s)", analysis_result.category)
            add_field_to_message("Descripción", analysis_result.description)
            add_field_to_message("LaTeX (Extraído)", analysis_result.latex, is_code=True)
            add_field_to_message("Derivación", analysis_result.derivation, is_code=True)
            add_field_to_message("Usos Comunes", analysis_result.uses)
            add_field_to_message("Variables", analysis_result.vars) # Era string
            add_field_to_message("Ecuaciones Similares", analysis_result.similars) # Era string
            
            # Debug info
            message_parts.append(f"\n\n<tg-spoiler><i>Debug Info:\nLLM Status: {html.escape(analysis_result.llm_analysis_status)}\nDB Status: {html.escape(analysis_result.database_status)}\nID: {analysis_result.equation_id}</i></tg-spoiler>")
            
            final_text = "\n".join(message_parts)
            # Truncar mensaje si es muy largo para Telegram
            if len(final_text) > 4096:
                final_text = final_text[:4090] + "\n(...)"
            
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_message.message_id,
                text=final_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )

            # Renderizar y enviar la imagen del LaTeX si se obtuvo
            if analysis_result.latex and (rendered_image_buffer := render_latex_to_image_bytes(analysis_result.latex, f"eq_{analysis_result.equation_id}.png")):
                try:
                    rendered_image_buffer.seek(0) # Reset buffer position
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=rendered_image_buffer,
                        caption="Ecuación (Renderizada de LaTeX)",
                        reply_to_message_id=processing_message.message_id # Responder al mensaje original de "procesando"
                    )
                    logger.info(f"Imagen renderizada de LaTeX enviada para: {analysis_result.latex[:30]}...")
                except Exception as e_send_photo:
                    logger.error(f"Error enviando imagen renderizada de LaTeX: {e_send_photo}", exc_info=True)
                finally:
                    if rendered_image_buffer: rendered_image_buffer.close()
            else:
                logger.warning(f"No se pudo renderizar o no había LaTeX para la imagen: {analysis_result.latex[:50] if analysis_result.latex else 'N/A'}")

        elif analysis_result and not analysis_result.latex: # Hubo un resultado de process_equation pero sin LaTeX
             await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_message.message_id,
                text=f"😕 No pude extraer una ecuación LaTeX de la imagen. Estado: {analysis_result.llm_analysis_status}"
            )
        else: # Fallo completo en process_equation
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_message.message_id,
                text="Lo siento, ocurrió un error al procesar la imagen de la ecuación. Por favor, inténtalo de nuevo más tarde."
            )

    except Exception as e:
        logger.error(f"Error crítico en handle_photo: {e}", exc_info=True)
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_message.message_id,
                text="🤕 ¡Ups! Algo salió muy mal. El equipo de desarrollo (o sea, Vaskias) ha sido notificado."
            )
        except: # Fallback si editar el mensaje falla
            await context.bot.send_message(chat_id=chat_id, text="🤕 ¡Ups! Algo salió muy mal.")


# --- Error Handler (Permanece igual) ---
async def error_handler(update: object | None, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Excepción no controlada por la aplicación: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Se encontró un error inesperado. Ya estoy trabajando en ello.")
        except Exception as e_err_handler:
            logger.error(f"Error dentro del error_handler al enviar mensaje: {e_err_handler}")


# --- Configuración App Bot Telegram y FastAPI Lifespan (Permanece igual) ---
telegram_app: Application | None = None
polling_task: asyncio.Task | None = None

if TELEGRAM_BOT_TOKEN:
    try:
        telegram_app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
        # Opcional: configurar connection_pool_size si se esperan muchas peticiones concurrentes del bot
        # telegram_app_builder.connection_pool_size(10)
        telegram_app = telegram_app_builder.build()

        telegram_app.add_handler(CommandHandler("start", start_command))
        telegram_app.add_handler(CommandHandler("help", help_command))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
        telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        telegram_app.add_error_handler(error_handler)
        logger.info("Aplicación de Telegram configurada exitosamente.")
    except Exception as e:
        logger.error(f"Error crítico al inicializar la aplicación de Telegram: {e}", exc_info=True)
        telegram_app = None # Asegurar que no se intente usar si falla la inicialización

async def run_telegram_polling():
    if not telegram_app or not hasattr(telegram_app, 'updater') or not telegram_app.updater:
        logger.error("Aplicación de Telegram o su Updater no están disponibles. No se puede iniciar polling.")
        return

    logger.info("Iniciando tarea de polling de Telegram...")
    try:
        logger.info("Llamando a telegram_app.initialize() (necesario antes de start_polling).")
        await telegram_app.initialize()
        logger.info("telegram_app.initialize() completado.")

        logger.info("Iniciando telegram_app.updater.start_polling()...")
        await telegram_app.updater.start_polling(
            allowed_updates=Update.ALL_TYPES, # Considera ser más específico si no necesitas todos los tipos
            drop_pending_updates=True # Ignora actualizaciones viejas al iniciar
        )
        logger.info("Polling de Telegram iniciado y activo.")

        # Bucle para mantener la tarea viva mientras el updater está corriendo.
        # La cancelación vendrá del lifespan de FastAPI.
        while telegram_app.updater and telegram_app.updater.running:
            await asyncio.sleep(1) # Ajusta el sleep si es necesario, 1s es un buen compromiso
        logger.info("El bucle de polling (while updater.running) ha terminado o el updater ya no está disponible.")

    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        logger.info("Polling de Telegram interrumpido o cancelado.")
    except Exception as e:
        logger.error(f"Error excepcional durante la ejecución del polling de Telegram: {e}", exc_info=True)
    finally:
        logger.info("Bloque finally de run_telegram_polling alcanzado.")
        if telegram_app and hasattr(telegram_app, 'updater') and telegram_app.updater and telegram_app.updater.running:
            logger.info("Deteniendo updater de Telegram (desde finally de run_telegram_polling)...")
            await telegram_app.updater.stop()
        if telegram_app:
            logger.info("Ejecutando telegram_app.shutdown() (desde finally de run_telegram_polling)...")
            await telegram_app.shutdown()
        logger.info("Tarea de polling (run_telegram_polling) finalizada completamente.")


from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app_lifespan: FastAPI): # app_lifespan es la instancia de FastAPI
    global polling_task
    logger.info("FastAPI lifespan [STARTUP]: Iniciando tareas de fondo...")
    if telegram_app: # Solo intentar polling si la app de telegram se configuró bien
        polling_task = asyncio.create_task(run_telegram_polling())
        logger.info(f"Tarea de polling de Telegram creada: {polling_task}")
        await asyncio.sleep(2) # Dar un momento para que la tarea inicie y potencialmente falle rápido
        if polling_task.done():
            try:
                exc = polling_task.exception()
                if exc:
                     logger.error(f"La tarea de polling falló inmediatamente después de crearla: {exc}", exc_info=exc)
            except asyncio.CancelledError:
                 logger.info("La tarea de polling fue cancelada inmediatamente (esto es raro aquí).")
            except asyncio.InvalidStateError: # Si ya se obtuvo la excepción
                 logger.info("La tarea de polling está en un estado inválido (probablemente cancelada).")
    else:
        logger.warning("Aplicación Telegram no inicializada. No se iniciará la tarea de polling.")
    
    yield # Aquí es donde la aplicación FastAPI se ejecuta
    
    logger.info("FastAPI lifespan [SHUTDOWN]: Deteniendo tareas de fondo...")
    if polling_task and not polling_task.done():
        logger.info("Cancelando tarea de polling de Telegram...")
        polling_task.cancel()
        try:
            await polling_task # Esperar a que la tarea de polling termine después de ser cancelada
        except asyncio.CancelledError:
            logger.info("Tarea de polling de Telegram cancelada exitosamente durante el shutdown.")
        except Exception as e_task_shutdown:
            logger.error(f"Error esperando la cancelación/finalización de la tarea de polling: {e_task_shutdown}", exc_info=True)
    elif polling_task and polling_task.done():
        logger.info("La tarea de polling ya había finalizado antes del shutdown del lifespan.")

    # Doble chequeo por si el finally de run_telegram_polling no se ejecutó completamente
    if telegram_app and hasattr(telegram_app, 'updater') and telegram_app.updater and telegram_app.updater.running:
        logger.warning("Updater de Telegram seguía corriendo durante el shutdown del lifespan. Intentando detener de nuevo.")
        await telegram_app.updater.stop()
    if telegram_app and hasattr(telegram_app, 'shutdown') and callable(telegram_app.shutdown): # Asegurar que es llamable
        logger.warning("Aplicación PTB podría seguir activa durante el shutdown del lifespan. Intentando shutdown de nuevo.")
        await telegram_app.shutdown()
    logger.info("Lifespan de FastAPI y limpieza de tareas de fondo finalizado.")

# Asignar el lifespan a la aplicación FastAPI
app.router.lifespan_context = lifespan


# --- Endpoint Webhook y Raíz (Sin cambios, pero el webhook no se usa activamente si se hace polling) ---
# Si usas polling, este endpoint de webhook no recibirá nada de Telegram,
# pero puede ser útil si alguna vez cambias a webhooks.
TELEGRAM_WEBHOOK_SECRET_PATH = f"/{TELEGRAM_BOT_TOKEN.split(':')[0]}_wh" if TELEGRAM_BOT_TOKEN and ':' in TELEGRAM_BOT_TOKEN else "/tg_wh_raphael_fb"
@app.post(TELEGRAM_WEBHOOK_SECRET_PATH)
async def telegram_webhook_endpoint(request: Request): # Renombrado para claridad
    if not telegram_app or not hasattr(telegram_app, 'bot'):
        logger.warning("Intento de webhook pero la app de Telegram no está configurada.")
        return Response(content="Bot no configurado.", status_code=503) # Service Unavailable
    try:
        update_data = await request.json()
        update = Update.de_json(update_data, telegram_app.bot)
        logger.info("Webhook de Telegram recibido (si está configurado).")
        await telegram_app.process_update(update)
        return Response(status_code=200)
    except json.JSONDecodeError:
        logger.error("Error decodificando JSON del webhook de Telegram.")
        return Response(content="Cuerpo JSON inválido.", status_code=400)
    except Exception as e:
        logger.error(f"Error procesando webhook de Telegram: {e}", exc_info=True)
        return Response(content="Error interno procesando webhook.", status_code=500)

@app.get("/")
async def root():
    return {"message": f"Raphael API (v{app.version}) funcionando. El bot de Telegram debería estar activo si el token está configurado."}

# --- Ejecución Directa (Polling Bloqueante si se ejecuta el script directamente) ---
# Esta sección es útil para desarrollo local si no usas un servidor ASGI como Uvicorn externamente.
# Para producción en App Service, Gunicorn+Uvicorn se encargarán de ejecutar la app FastAPI,
# y el lifespan se encargará de iniciar el polling de Telegram.
if __name__ == "__main__":
    # Este bloque __main__ no es estrictamente necesario si siempre vas a ejecutar con Uvicorn/Gunicorn.
    # Pero es útil para pruebas rápidas locales.
    if telegram_app:
        logger.info("Ejecución directa del script (__main__): Iniciando polling de Telegram de forma bloqueante...")
        # Nota: En este modo, FastAPI no se sirve a menos que lo hagas explícitamente en otro hilo,
        # o uses run_polling en un hilo y uvicorn.run en el principal.
        # El lifespan de FastAPI no se ejecuta en este modo `telegram_app.run_polling()`.
        # Para desarrollo local combinado, es mejor usar `uvicorn main:app --reload` y dejar
        # que el lifespan maneje el polling.
        try:
            # `run_polling` es bloqueante y maneja su propio `initialize` y `shutdown`.
            telegram_app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
            logger.info("Polling directo de Telegram detenido.")
        except KeyboardInterrupt:
            logger.info("Polling directo de Telegram detenido por el usuario (KeyboardInterrupt).")
        except Exception as e:
            logger.error(f"Error crítico durante el polling directo de Telegram: {e}", exc_info=True)
    else:
        logger.error("Error Crítico: Aplicación Telegram no inicializada. Polling directo no puede iniciarse.")
    
    # Si quisieras correr FastAPI también desde aquí (no recomendado si ya usas lifespan para polling):
    # import uvicorn
    # logger.info("Para ejecutar FastAPI y el bot (con polling en lifespan), usa: uvicorn main:app --host 0.0.0.0 --port 8000")