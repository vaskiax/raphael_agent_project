# -*- coding: utf-8 -*-
# Versión 0.6.5 - Corrección SyntaxError y lifespan con Application.run_polling
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File
import asyncio
import os
import uuid
from pydantic import BaseModel, ValidationError, Field
import logging
import json
import re
import io
from PIL import Image
import html
from contextlib import asynccontextmanager

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure, InvalidName

import httpx

from dotenv import load_dotenv

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy import latex as sympy_latex
    sympy_available = True
except ImportError:
    sympy_available = False
    logging.warning("SymPy no encontrado. La normalización de LaTeX será básica.")
    parse_latex = None
    sympy_latex = None

import telegram
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, TypeHandler, filters
)
from telegram.constants import ParseMode

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.INFO)
# Niveles de DEBUG para python-telegram-bot
logging.getLogger("telegram.ext").setLevel(logging.DEBUG)
logging.getLogger("telegram.bot").setLevel(logging.DEBUG)
logging.getLogger("telegram.request").setLevel(logging.DEBUG)
logger = logging.getLogger("main_raphael_core") # Nombre de logger más específico

app = FastAPI(title="Raphael Agent API & Telegram Bot", version="0.6.5")

MONGO_CONNECTION_STRING = os.getenv("COSMOS_MONGO_CONNECTION_STRING")
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "raphaeldb")
COLLECTION_NAME = os.getenv("COSMOS_COLLECTION_NAME", "equations")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAPHEYE_VISION_FUNCTION_URL = os.getenv("RAPHEYE_VISION_FUNCTION_URL")

logger.info(f"COSMOS_MONGO_CONNECTION_STRING loaded: {'Yes' if MONGO_CONNECTION_STRING else 'NO'}")
logger.info(f"TELEGRAM_BOT_TOKEN loaded: {'Yes' if TELEGRAM_BOT_TOKEN else 'NO'}")
logger.info(f"RAPHEYE_VISION_FUNCTION_URL loaded: {'Yes' if RAPHEYE_VISION_FUNCTION_URL else 'NO'}")

if not TELEGRAM_BOT_TOKEN:
    logger.critical("¡¡¡ TELEGRAM_BOT_TOKEN no está configurado!!! El bot no funcionará.")
if not RAPHEYE_VISION_FUNCTION_URL:
    logger.critical("¡¡¡ RAPHEYE_VISION_FUNCTION_URL no está configurada!!! El análisis de imágenes no funcionará.")

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
    logger.warning("COSMOS_MONGO_CONNECTION_STRING no configurada. Operaciones de DB desactivadas.")

ALLOWED_CATEGORIES = [
    "mecánica clasica", "mecánica cuantica", "algebra lineal", "cálculo",
    "ecuaciones diferenciales parciales", "ecuaciones diferenciales ordinarias",
    "estadística y probabilidad", "termodinámica", "relatividad",
    "series (taylor, fourier, laurent, etc...)", "genericas", "óptica", "electromagnetismo"
]
class LLMAnalysisData(BaseModel):
    latex_extracted_from_image: str | None = None; name: str | None = None; category: str | None = None
    description: str | None = None; derivation: str | None = None; uses: str | None = None
    vars: str | None = None; similars: str | None = None
class EquationAnalysis(BaseModel):
    equation_id: str; latex: str; name: str | None = None; category: str | None = None
    description: str | None = None; derivation: str | None = None; uses: str | None = None
    vars: str | None = None; similars: str | None = None; llm_analysis_status: str
    database_status: str; normalized_latex_key: str | None = None

def normalize_latex(latex_str: str) -> str | None:
    if not latex_str: return None
    normalized = re.sub(r'\s+', ' ', latex_str).strip()
    if normalized.startswith("$$") and normalized.endswith("$$"): normalized = normalized[2:-2].strip()
    if normalized.startswith("\\[") and normalized.endswith("\\]"): normalized = normalized[2:-2].strip()
    if normalized.startswith("$") and normalized.endswith("$"): normalized = normalized[1:-1].strip()
    if sympy_available and parse_latex and sympy_latex:
        try: sympy_expr = parse_latex(normalized); normalized_sympy = sympy_latex(sympy_expr, mode='inline', mul_symbol='dot'); return normalized_sympy.strip()
        except Exception: return normalized
    return normalized
async def find_equation_by_normalized_latex(normalized_latex_key: str) -> dict | None:
    if db_collection is None or not normalized_latex_key: return None
    try: return db_collection.find_one({"normalized_latex_key": normalized_latex_key})
    except Exception as e: logger.error(f"Error buscando en DB: {e}", exc_info=True); return None

async def call_rapheye_vision_azure_function(image_bytes: bytes, image_mime_type: str) -> LLMAnalysisData | None:
    if not RAPHEYE_VISION_FUNCTION_URL: logger.error("RAPHEYE_VISION_FUNCTION_URL no configurada."); return None
    if not image_bytes: logger.error("Bytes de imagen vacíos para Azure Function."); return None
    headers = {"Content-Type": image_mime_type}; timeout_config = httpx.Timeout(190.0, connect=15.0)
    response_obj_for_debug = None 
    try:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            logger.info(f"Enviando imagen a Azure Function: {RAPHEYE_VISION_FUNCTION_URL[:80]}...")
            response_obj_for_debug = await client.post(RAPHEYE_VISION_FUNCTION_URL, content=image_bytes, headers=headers)
        response_obj_for_debug.raise_for_status()
        response_json = response_obj_for_debug.json()
        logger.info("Respuesta JSON de Azure Function.")
        try: return LLMAnalysisData(**response_json)
        except ValidationError as e_val: logger.error(f"Error Pydantic respuesta Azure Function: {e_val}"); logger.debug(f"JSON fallido: {response_json}"); return None
    except httpx.HTTPStatusError as e_http:
        logger.error(f"Error HTTP {e_http.response.status_code} llamando Azure Function: {e_http.response.text[:500]}")
        return None
    except httpx.RequestError as e_req: 
        logger.error(f"Error red/solicitud llamando Azure Function: {type(e_req).__name__} - {e_req}")
        return None
    except json.JSONDecodeError as e_json: # <-- BLOQUE CORREGIDO
        logger.error(f"Error al decodificar JSON de Azure Function: {e_json}")
        response_text_for_debug = "No disponible"
        if response_obj_for_debug and hasattr(response_obj_for_debug, 'text'):
            response_text_for_debug = response_obj_for_debug.text[:500]
        logger.debug(f"Respuesta cruda (si disponible) que causó error JSON: {response_text_for_debug}")
        return None
    except Exception as e_general: 
        logger.error(f"Error inesperado llamando Azure Function: {e_general}", exc_info=True)
        return None

async def save_analysis_to_mongo(analysis_data: EquationAnalysis, normalized_latex_key: str) -> bool: # Sin cambios
    if db_collection is None: logger.warning("DB no configurada."); return False
    try:
        doc_to_save = analysis_data.model_dump() if hasattr(analysis_data, 'model_dump') else analysis_data.dict()
        doc_to_save['_id'] = analysis_data.equation_id; doc_to_save['normalized_latex_key'] = normalized_latex_key
        res = db_collection.replace_one({'_id': analysis_data.equation_id}, doc_to_save, upsert=True)
        return res.acknowledged
    except Exception as e: logger.error(f"Error guardando DB: {e}", exc_info=True); return False

def render_latex_to_image_bytes(latex_str: str | None, filename: str = "equation.png") -> io.BytesIO | None: # Sin cambios
    if not latex_str or not latex_str.strip(): logger.warning(f"Render LaTeX inválido/vacío: {filename}"); return None
    txt = latex_str.strip();
    if not (txt.startswith('$') and txt.endswith('$')) and not txt.startswith('\\['): txt = f"${txt}$"
    fig = None
    try:
        fig_height = max(2.5, len(txt.splitlines()) * 0.6 + 1.0); fig_width = 10
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200); fig.patch.set_facecolor('white')
        ax.text(0.5, 0.5, txt, fontsize=18, ha='center', va='center', wrap=True); ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2); buf.seek(0)
        logger.info(f"Renderizado LaTeX a PNG ok: {filename}"); return buf
    except Exception as e: logger.error(f"Error renderizando LaTeX '{txt[:30]}...': {e}", exc_info=True); return None
    finally:
        if fig: plt.close(fig)

async def process_equation(image_bytes: bytes, image_mime_type: str, filename: str = "telegram_image") -> EquationAnalysis | None: # Sin cambios mayores
    llm_data = await call_rapheye_vision_azure_function(image_bytes, image_mime_type)
    if not llm_data or not llm_data.latex_extracted_from_image:
        logger.error("Fallo al obtener análisis de Azure Function o no se extrajo LaTeX.")
        return EquationAnalysis(equation_id=str(uuid.uuid4()), latex="", llm_analysis_status="Failed (Azure Function error or no LaTeX)", database_status="N/A - Analysis Failed")
    extracted_latex = llm_data.latex_extracted_from_image; normalized_key = normalize_latex(extracted_latex) or extracted_latex
    if db_collection is not None and (existing_doc_dict := await find_equation_by_normalized_latex(normalized_key)):
        try:
            data_for_model = {**existing_doc_dict}; data_for_model['latex'] = data_for_model.get('latex_extracted_from_image', data_for_model.get('latex'))
            data_for_model['database_status'] = "Retrieved from DB"; data_for_model['llm_analysis_status'] = data_for_model.get('llm_analysis_status', 'Retrieved (Original status unknown)')
            if 'equation_id' not in data_for_model and '_id' in data_for_model: data_for_model['equation_id'] = str(data_for_model['_id'])
            model_fields = EquationAnalysis.model_fields.keys(); cleaned_data_for_model = {k: v for k, v in data_for_model.items() if k in model_fields}
            logger.info(f"Ecuación encontrada en DB con normalized_key: {normalized_key}"); return EquationAnalysis(**cleaned_data_for_model)
        except ValidationError as e: logger.error(f"Error Pydantic al cargar desde DB: {e}. Datos: {existing_doc_dict}. Re-analizando.")
        except Exception as e: logger.error(f"Error inesperado procesando doc DB: {e}. Re-analizando.", exc_info=True)
    analysis = EquationAnalysis(
        equation_id=str(uuid.uuid4()), latex=extracted_latex, name=llm_data.name, category=llm_data.category, description=llm_data.description,
        derivation=llm_data.derivation, uses=llm_data.uses, vars=llm_data.vars, similars=llm_data.similars,
        llm_analysis_status="Success (via Azure Function)", database_status="Save Attempt Pending", normalized_latex_key=normalized_key)
    if db_collection is not None:
        if await save_analysis_to_mongo(analysis, normalized_key): analysis.database_status = "Saved Successfully (New Entry)"; logger.info(f"Análisis guardado en DB para key: {normalized_key}")
        else: analysis.database_status = "Save Failed"; logger.warning(f"Fallo al guardar análisis en DB para key: {normalized_key}")
    elif db_collection is None: analysis.database_status = "Skipped (DB Unavailable)"
    return analysis

async def generic_update_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: # Sin cambios
    logger.info(f"--- GENERIC UPDATE RECEIVED --- Update Type: {type(update)}")
    if update.message: logger.info(f"Generic Update - Message details: ID={update.message.message_id}, Text='{update.message.text}', AttachmentType={update.message.effective_attachment if update.message.effective_attachment else 'N/A'}")
    elif update.callback_query: logger.info(f"Generic Update - CallbackQuery: Data='{update.callback_query.data}'")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: await update.message.reply_html(f"¡Hola {update.effective_user.mention_html()}! Envíame foto de ecuación.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: await update.message.reply_html("Envíame foto de ecuación matemática. <b>Consejos:</b> Imagen clara.")
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: await update.message.reply_text("Por favor, envíame FOTO de ecuación.")
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: # Sin cambios mayores
    logger.info(f"--- handle_photo INVOCADO --- Chat ID: {update.effective_chat.id}")
    if not update.message or not update.message.photo: logger.warning("Mensaje sin foto en handle_photo."); return
    chat_id = update.effective_chat.id; processing_message = await context.bot.send_message(chat_id=chat_id, text="📸 Analizando ecuación, espera...")
    try:
        photo_file = await update.message.photo[-1].get_file(); image_bytes_io = io.BytesIO(); await photo_file.download_to_memory(image_bytes_io)
        image_bytes = image_bytes_io.getvalue(); image_format = "JPEG"; image_mime_type = "image/jpeg"
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)); image_format = pil_image.format or "JPEG"
            mime_type_map = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "BMP": "image/bmp"}
            image_mime_type = mime_type_map.get(image_format.upper(), "image/jpeg")
            logger.info(f"Imagen recibida en handle_photo. Formato: {image_format}, MIME: {image_mime_type}")
        except Exception as e_pil: logger.error(f"Error Pillow en handle_photo: {e_pil}.")
        analysis_result = await process_equation(image_bytes, image_mime_type, filename=f"tg_{photo_file.file_id}.{image_format.lower()}")
        if analysis_result and analysis_result.latex:
            message_parts = ["✅ <b>¡Análisis Completado!</b>"];
            if analysis_result.database_status == "Retrieved from DB": message_parts.append("<i>(Info de DB)</i>")
            def add_field(lbl, val, code=False):
                if val and str(val).strip(): esc_val = html.escape(str(val).strip()); message_parts.append(f"\n<b>{html.escape(lbl)}:</b>{('<code>'+esc_val+'</code>') if code else (' '+esc_val)}")
            add_field("Nombre", analysis_result.name); add_field("Categoría(s)", analysis_result.category); add_field("Descripción", analysis_result.description)
            add_field("LaTeX (Extraído)", analysis_result.latex, True); add_field("Derivación", analysis_result.derivation, True)
            add_field("Usos Comunes", analysis_result.uses); add_field("Variables", analysis_result.vars); add_field("Ecuaciones Similares", analysis_result.similars)
            message_parts.append(f"\n\n<tg-spoiler><i>Debug: LLM:{html.escape(analysis_result.llm_analysis_status)},DB:{html.escape(analysis_result.database_status)},ID:{analysis_result.equation_id}</i></tg-spoiler>")
            final_text = "\n".join(message_parts);
            if len(final_text) > 4096: final_text = final_text[:4090] + "\n(...)"
            await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_message.message_id, text=final_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            if analysis_result.latex and (img_buf := render_latex_to_image_bytes(analysis_result.latex, f"eq_{analysis_result.equation_id}.png")):
                try: img_buf.seek(0); await context.bot.send_photo(chat_id=chat_id, photo=img_buf, caption="Ecuación (Renderizada)", reply_to_message_id=processing_message.message_id); logger.info(f"Img renderizada enviada: {analysis_result.latex[:30]}...")
                except Exception as e_send_photo: logger.error(f"Error enviando img renderizada: {e_send_photo}", exc_info=True)
                finally:
                    if img_buf: img_buf.close()
            else: logger.warning(f"No se renderizó/envió img para LaTeX: {analysis_result.latex[:50] if analysis_result.latex else 'N/A'}")
        elif analysis_result and not analysis_result.latex: await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_message.message_id, text=f"😕 No pude extraer LaTeX. Estado: {analysis_result.llm_analysis_status}")
        else: await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_message.message_id, text="Error procesando imagen.")
    except Exception as e:
        logger.error(f"Error crítico en handle_photo: {e}", exc_info=True)
        try: await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_message.message_id, text="🤕 ¡Ups! Algo salió muy mal.")
        except: await context.bot.send_message(chat_id=chat_id, text="🤕 ¡Ups! Algo salió muy mal.")

async def error_handler(update: object | None, context: ContextTypes.DEFAULT_TYPE) -> None: # Sin cambios
    logger.error(f"Excepción no controlada: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try: await context.bot.send_message(chat_id=update.effective_chat.id, text="Error inesperado.")
        except Exception as e_err_handler: logger.error(f"Error en error_handler al enviar msg: {e_err_handler}")

telegram_app: Application | None = None
polling_task: asyncio.Task | None = None

if TELEGRAM_BOT_TOKEN: # Sin cambios
    try:
        custom_http_request = telegram.request.HTTPXRequest(
            connect_timeout=10.0, read_timeout=60.0, write_timeout=10.0, pool_timeout=5.0
        )
        telegram_app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
        telegram_app_builder.request(custom_http_request)
        telegram_app = telegram_app_builder.build()
        telegram_app.add_handler(TypeHandler(Update, generic_update_handler), group=-1)
        telegram_app.add_handler(CommandHandler("start", start_command))
        telegram_app.add_handler(CommandHandler("help", help_command))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
        telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        telegram_app.add_error_handler(error_handler)
        logger.info("Aplicación de Telegram configurada exitosamente.")
    except Exception as e:
        logger.error(f"Error crítico al inicializar la aplicación de Telegram: {e}", exc_info=True)
        telegram_app = None

# --- ELIMINAR la función run_telegram_polling personalizada ---
# async def run_telegram_polling():
#    ... (ESTA FUNCIÓN DEBE SER ELIMINADA COMPLETAMENTE)

# --- USAR ESTE LIFESPAN SIMPLIFICADO ---
@asynccontextmanager
async def lifespan(app_lifespan_instance: FastAPI):
    global polling_task
    logger.info("FastAPI lifespan [STARTUP]: Iniciando.")
    if TELEGRAM_BOT_TOKEN and telegram_app:
        logger.info("Intentando iniciar telegram_app.run_polling() directamente como tarea...")
        try:
            polling_task = asyncio.create_task(telegram_app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                timeout=30, 
            ))
            task_name = polling_task.get_name() if hasattr(polling_task, 'get_name') else "PollingTaskAppRunPolling"
            logger.info(f"Tarea de polling de Telegram (usando Application.run_polling) creada: {task_name}")
            
            await asyncio.sleep(5) 
            if polling_task.done():
                logger.error(f"¡La tarea de polling ({task_name}) finalizó/falló inmediatamente!")
                try:
                    exc = polling_task.exception()
                    if exc: logger.error(f"Excepción de {task_name} inmediata: {type(exc).__name__} - {exc}", exc_info=True)
                    else: logger.info(f"Tarea {task_name} finalizó inmediatamente sin excepción (raro).")
                except asyncio.CancelledError:
                    logger.info(f"Tarea {task_name} fue cancelada inmediatamente.")
                except Exception as e_poll_check:
                    logger.error(f"Error verificando tarea {task_name}: {e_poll_check}", exc_info=True)
            else:
                logger.info(f"Tarea de polling ({task_name}) parece estar corriendo después de 5s.")

        except Exception as e_lifespan_startup:
            logger.error(f"Error en lifespan startup al iniciar Telegram con Application.run_polling: {e_lifespan_startup}", exc_info=True)
    else:
        logger.warning("Token de Telegram o app no configurados. No se inicia polling.")
    
    yield 
    
    logger.info("FastAPI lifespan [SHUTDOWN]: Deteniendo.")
    if polling_task and not polling_task.done():
        task_name_shutdown = polling_task.get_name() if hasattr(polling_task, 'get_name') else "PollingTaskAppRunPolling"
        logger.info(f"Cancelando tarea de polling ({task_name_shutdown})...")
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            logger.info(f"Tarea de polling ({task_name_shutdown}) cancelada exitosamente.")
        except Exception as e_shutdown:
            logger.error(f"Excepción durante la espera de cancelación de {task_name_shutdown}: {type(e_shutdown).__name__} - {e_shutdown}", exc_info=True)
    
    if telegram_app and hasattr(telegram_app, 'shutdown') and callable(telegram_app.shutdown):
        logger.info("Llamando a telegram_app.shutdown() explícitamente en el shutdown del lifespan.")
        try:
            await telegram_app.shutdown()
            logger.info("telegram_app.shutdown() completado.")
        except Exception as e_explicit_shutdown:
            logger.error(f"Error en el shutdown explícito de telegram_app: {e_explicit_shutdown}", exc_info=True)
    logger.info("Lifespan de FastAPI finalizado.")

app.router.lifespan_context = lifespan # Asegurar que esta línea esté DESPUÉS de definir lifespan

TELEGRAM_WEBHOOK_SECRET_PATH = f"/{TELEGRAM_BOT_TOKEN.split(':')[0]}_wh" if TELEGRAM_BOT_TOKEN and ':' in TELEGRAM_BOT_TOKEN else "/tg_wh_raphael_fb"
@app.post(TELEGRAM_WEBHOOK_SECRET_PATH)
async def telegram_webhook_endpoint(request: Request): # Sin cambios
    if not telegram_app or not hasattr(telegram_app, 'bot'):
        logger.warning("Intento de webhook pero la app de Telegram no está configurada.")
        return Response(content="Bot no configurado.", status_code=503)
    try:
        update_data = await request.json(); update = Update.de_json(update_data, telegram_app.bot)
        logger.info("Webhook de Telegram recibido (si está configurado)."); await telegram_app.process_update(update)
        return Response(status_code=200)
    except json.JSONDecodeError: logger.error("Error decodificando JSON del webhook."); return Response(content="Cuerpo JSON inválido.", status_code=400)
    except Exception as e: logger.error(f"Error procesando webhook: {e}", exc_info=True); return Response(content="Error interno procesando webhook.", status_code=500)

@app.get("/")
async def root(): # Sin cambios
    return {"message": f"Raphael API (v{app.version}) funcionando. El bot de Telegram debería estar activo si el token está configurado."}

if __name__ == "__main__": # Sin cambios
    if telegram_app:
        logger.info("Ejecución directa: Para FastAPI y polling en lifespan, usa: uvicorn main:app --reload")
        logger.info("Iniciando polling bloqueante (solo para pruebas directas del bot)...")
        try:
            telegram_app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
            logger.info("Polling directo de Telegram detenido.")
        except KeyboardInterrupt: logger.info("Polling directo detenido por usuario.")
        except Exception as e: logger.error(f"Error crítico polling directo: {e}", exc_info=True)
    else:
        logger.error("Error Crítico: Aplicación Telegram no inicializada. Polling directo no puede iniciarse.")