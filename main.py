# -*- coding: utf-8 -*-
# Versión 0.10.1 - Webhooks - Debugging de Endpoint y Logs PTB Detallados
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
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
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure, InvalidName

import httpx # Para hacer la llamada setWebhook y para Rapheye

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

import telegram # Importar para telegram.error
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, TypeHandler, filters
)
from telegram.constants import ParseMode

# --- Configuración de Logging ---
load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.INFO)
# MODIFICACIÓN: Niveles de DEBUG para python-telegram-bot
logging.getLogger("telegram.ext").setLevel(logging.DEBUG)
logging.getLogger("telegram.bot").setLevel(logging.DEBUG)
logging.getLogger("telegram.request").setLevel(logging.DEBUG)
logging.getLogger("telegram.ext.dispatcher").setLevel(logging.DEBUG) # Muy importante
logger = logging.getLogger("raphael_core_webhook_v0101") # VERSIÓN LOGGER ACTUALIZADA

# --- FastAPI App ---
app = FastAPI(title="Raphael Agent API & Telegram Bot (Webhooks)", version="0.10.1") # VERSIÓN APP ACTUALIZADA

# --- Variables Globales de Configuración (desde .env) ---
MONGO_CONNECTION_STRING = os.getenv("COSMOS_MONGO_CONNECTION_STRING")
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "raphaeldb")
COLLECTION_NAME = os.getenv("COSMOS_COLLECTION_NAME", "equations")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAPHEYE_VISION_FUNCTION_URL = os.getenv("RAPHEYE_VISION_FUNCTION_URL")

WEBHOOK_URL_BASE = os.getenv("WEBHOOK_URL_BASE") 
WEBHOOK_SECRET_PATH_SUFFIX = "webhook_updates" 

logger.info(f"COSMOS_MONGO_CONNECTION_STRING loaded: {'Yes' if MONGO_CONNECTION_STRING else 'NO'}")
logger.info(f"TELEGRAM_BOT_TOKEN loaded: {'Yes' if TELEGRAM_BOT_TOKEN else 'NO'}")
logger.info(f"RAPHEYE_VISION_FUNCTION_URL loaded: {'Yes' if RAPHEYE_VISION_FUNCTION_URL else 'NO'}")
logger.info(f"WEBHOOK_URL_BASE loaded: {'Yes' if WEBHOOK_URL_BASE else 'NO - ¡NECESARIO PARA WEBHOOKS!'}")

if not TELEGRAM_BOT_TOKEN:
    logger.critical("¡TELEGRAM_BOT_TOKEN no está configurado! El bot no funcionará.")
if not WEBHOOK_URL_BASE and os.getenv("RUN_MODE", "dev") != "local_no_webhook": 
    logger.critical("¡WEBHOOK_URL_BASE no está configurado! El webhook no se registrará con Telegram.")

# --- Variables Globales para estado de la aplicación ---
db_client: pymongo.MongoClient | None = None
db_collection: pymongo.collection.Collection | None = None
ptb_application: Application | None = None

# --- Modelos Pydantic (Tu lógica existente - sin cambios) ---
class LLMAnalysisData(BaseModel):
    latex_extracted_from_image: str | None = None; name: str | None = None; category: str | None = None
    description: str | None = None; derivation: str | None = None; uses: str | None = None
    vars: str | None = None; similars: str | None = None
class EquationAnalysis(BaseModel):
    equation_id: str; latex: str; name: str | None = None; category: str | None = None
    description: str | None = None; derivation: str | None = None; uses: str | None = None
    vars: str | None = None; similars: str | None = None; llm_analysis_status: str
    database_status: str; normalized_latex_key: str | None = None
class TaskContext(BaseModel):
    chat_id: int; user_id: int; message_id: int | None = None
    processing_message_id: int | None = None; image_bytes: bytes | None = None
    image_mime_type: str | None = None; image_filename: str | None = None
    llm_analysis_data: LLMAnalysisData | None = None
    equation_analysis_db: EquationAnalysis | None = None
    normalized_latex_key: str | None = None; found_in_db: bool = False
    error_message: str | None = None
    class Config: arbitrary_types_allowed = True

# --- Funciones Auxiliares (Tu lógica existente - sin cambios) ---
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
    try: return await asyncio.to_thread(db_collection.find_one, {"normalized_latex_key": normalized_latex_key})
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
    except Exception as e_general: logger.error(f"Error inesperado llamando Azure Function: {e_general}", exc_info=True); return None

async def save_analysis_to_mongo(analysis_data: EquationAnalysis, normalized_latex_key: str) -> bool:
    if db_collection is None: logger.warning("DB no configurada, no se guardará el análisis."); return False
    try:
        doc_to_save = analysis_data.model_dump(); doc_to_save['_id'] = analysis_data.equation_id; doc_to_save['normalized_latex_key'] = normalized_latex_key
        result = await asyncio.to_thread(db_collection.replace_one, {'_id': analysis_data.equation_id}, doc_to_save, upsert=True)
        return result.acknowledged
    except Exception as e: logger.error(f"Error guardando DB: {e}", exc_info=True); return False

def render_latex_to_image_bytes(latex_str: str | None, filename: str = "equation.png") -> io.BytesIO | None:
    if not latex_str or not latex_str.strip(): logger.warning(f"Render LaTeX inválido/vacío: {filename}"); return None
    txt = latex_str.strip();
    if not (txt.startswith('$') and txt.endswith('$')) and not txt.startswith('\\['): txt = f"${txt}$"
    fig = None
    try:
        fig_height = max(1.5, len(txt.splitlines()) * 0.5 + 0.5); fig_width = 8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150); fig.patch.set_facecolor('white')
        ax.text(0.5, 0.5, txt, fontsize=16, ha='center', va='center', wrap=True); ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1); buf.seek(0)
        logger.info(f"Renderizado LaTeX a PNG ok: {filename}"); return buf
    except Exception as e: logger.error(f"Error renderizando LaTeX '{txt[:30]}...': {e}", exc_info=False); logger.debug("Traceback render:", exc_info=True); return None
    finally:
        if fig: plt.close(fig)

async def orchestrate_image_processing(task_ctx: TaskContext) -> TaskContext:
    logger.info(f"Orquestación iniciada para Chat ID {task_ctx.chat_id}, User ID {task_ctx.user_id}")
    if not (task_ctx.image_bytes and task_ctx.image_mime_type):
        task_ctx.error_message = "Datos de imagen incompletos en el contexto."
        logger.error(f"Orquestación: {task_ctx.error_message}")
        task_ctx.equation_analysis_db = EquationAnalysis( equation_id=str(uuid.uuid4()), latex="", llm_analysis_status=task_ctx.error_message, database_status="N/A - Image Data Missing" )
        return task_ctx
    logger.info("Orquestación: Llamando a Rapheye (Azure Function)...")
    task_ctx.llm_analysis_data = await call_rapheye_vision_azure_function(task_ctx.image_bytes, task_ctx.image_mime_type)
    if not task_ctx.llm_analysis_data or not task_ctx.llm_analysis_data.latex_extracted_from_image:
        task_ctx.error_message = "Fallo el análisis de Rapheye o no se extrajo LaTeX de la imagen."
        logger.error(f"Orquestación: {task_ctx.error_message}")
        task_ctx.equation_analysis_db = EquationAnalysis( equation_id=str(uuid.uuid4()), latex=task_ctx.llm_analysis_data.latex_extracted_from_image if task_ctx.llm_analysis_data else "", llm_analysis_status=task_ctx.error_message, database_status="N/A - LLM Analysis Failed" )
        return task_ctx
    extracted_latex = task_ctx.llm_analysis_data.latex_extracted_from_image
    logger.info(f"Orquestación: LaTeX extraído por Rapheye: '{extracted_latex[:100]}...'")
    task_ctx.normalized_latex_key = normalize_latex(extracted_latex)
    if not task_ctx.normalized_latex_key: task_ctx.normalized_latex_key = extracted_latex
    logger.info(f"Orquestación: Clave LaTeX Normalizada: '{task_ctx.normalized_latex_key[:100]}...'")
    if db_collection is not None and task_ctx.normalized_latex_key:
        logger.info(f"Orquestación: Buscando en DB por clave normalizada...")
        existing_doc_dict = await find_equation_by_normalized_latex(task_ctx.normalized_latex_key)
        if existing_doc_dict:
            logger.info(f"Orquestación: Ecuación encontrada en DB con ID '{existing_doc_dict.get('_id')}' por clave normalizada.")
            try:
                data_for_model = {**existing_doc_dict}
                if 'latex_extracted_from_image' in data_for_model and 'latex' not in data_for_model: data_for_model['latex'] = data_for_model.pop('latex_extracted_from_image')
                data_for_model['equation_id'] = str(data_for_model.pop('_id', uuid.uuid4()))
                data_for_model['database_status'] = "Retrieved from DB"
                data_for_model.setdefault('llm_analysis_status', 'Retrieved from DB (original status unknown)')
                model_fields = EquationAnalysis.model_fields.keys()
                cleaned_data_for_model = {k: v for k, v in data_for_model.items() if k in model_fields}
                task_ctx.equation_analysis_db = EquationAnalysis(**cleaned_data_for_model)
                task_ctx.found_in_db = True
                logger.info(f"Orquestación: Datos cargados desde DB para ecuación ID: {task_ctx.equation_analysis_db.equation_id}")
                return task_ctx
            except Exception as e_db_load: logger.error(f"Orquestación: Error procesando doc de DB: {e_db_load}. Se procederá a re-analizar.", exc_info=True)
        else: logger.info("Orquestación: No se encontró la ecuación en la DB por clave normalizada. Se creará una nueva entrada.")
    logger.info("Orquestación: Creando nuevo objeto de análisis para la ecuación.")
    current_equation_id = str(uuid.uuid4())
    task_ctx.equation_analysis_db = EquationAnalysis( equation_id=current_equation_id, latex=extracted_latex, name=task_ctx.llm_analysis_data.name, category=task_ctx.llm_analysis_data.category, description=task_ctx.llm_analysis_data.description, derivation=task_ctx.llm_analysis_data.derivation, uses=task_ctx.llm_analysis_data.uses, vars=task_ctx.llm_analysis_data.vars, similars=task_ctx.llm_analysis_data.similars, llm_analysis_status="Success (via Azure Rapheye)", database_status="Save Attempt Pending", normalized_latex_key=task_ctx.normalized_latex_key )
    if db_collection is not None:
        logger.info(f"Orquestación: Intentando guardar nuevo análisis en DB (ID: {current_equation_id})...")
        if await save_analysis_to_mongo(task_ctx.equation_analysis_db, task_ctx.normalized_latex_key): # type: ignore
            task_ctx.equation_analysis_db.database_status = "Saved Successfully (New Entry)"
            logger.info(f"Orquestación: Nuevo análisis guardado en DB con ID: {current_equation_id}")
        else:
            task_ctx.equation_analysis_db.database_status = "Save Failed"
            logger.warning(f"Orquestación: Fallo al guardar nuevo análisis en DB (ID: {current_equation_id}).")
    elif db_collection is None:
        task_ctx.equation_analysis_db.database_status = "Skipped (DB Unavailable)"
        logger.warning("Orquestación: Guardado en DB omitido porque la conexión no está disponible.")
    logger.info(f"Orquestación completada para Chat ID {task_ctx.chat_id}.")
    return task_ctx

# --- Handlers de Telegram ---
async def generic_update_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"--- GENERIC UPDATE HANDLER INVOCADO --- Update ID: {update.update_id}, Type: {type(update)}")
    if update.message: logger.info(f"Generic Update Details - Message ID: {update.message.message_id}, Chat ID: {update.message.chat.id}, User: {update.effective_user.username if update.effective_user else 'N/A'}, Text: '{update.message.text if update.message.text else '[No Text]'}', Attachment: {update.message.effective_attachment if update.message.effective_attachment else 'N/A'}")
    elif update.callback_query: logger.info(f"Generic Update Details - CallbackQuery: Data='{update.callback_query.data}', From User: {update.callback_query.from_user.username if update.callback_query.from_user else 'N/A'}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"--- START_COMMAND HANDLER INVOCADO --- User: {update.effective_user.username if update.effective_user else 'N/A'}, Chat ID: {update.effective_chat.id}")
    if update.message: await update.message.reply_html(f"¡Hola {update.effective_user.mention_html()}! Soy Raphael (webhook v0.10.1). Envíame una foto de una ecuación matemática.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"--- HELP_COMMAND HANDLER INVOCADO --- User: {update.effective_user.username if update.effective_user else 'N/A'}, Chat ID: {update.effective_chat.id}")
    if update.message: await update.message.reply_html("<b>Raphael Bot (Webhooks v0.10.1)</b>\nEnvíame una foto de una ecuación matemática para análisis.")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"--- HANDLE_TEXT_MESSAGE HANDLER INVOCADO --- User: {update.effective_user.username if update.effective_user else 'N/A'}, Chat ID: {update.effective_chat.id}")
    if update.message: await update.message.reply_text("Por favor, envíame una FOTO de una ecuación. No proceso texto directamente (v0.10.1).")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"--- HANDLE_PHOTO HANDLER INVOCADO --- User: {update.effective_user.username if update.effective_user else 'N/A'}, Chat ID: {update.effective_chat.id}")
    if not (update.message and update.message.photo and update.effective_chat and update.effective_user):
        logger.warning("handle_photo: Mensaje de foto incompleto.")
        return
    chat_id = update.effective_chat.id; user_id = update.effective_user.id; message_id = update.message.message_id
    processing_msg = await context.bot.send_message(chat_id=chat_id, text="📸 Analizando tu ecuación (webhook v0.10.1)...")
    task_ctx = TaskContext(chat_id=chat_id, user_id=user_id, message_id=message_id, processing_message_id=processing_msg.message_id)
    try:
        photo_file = await update.message.photo[-1].get_file(); image_bytes_io = io.BytesIO(); await photo_file.download_to_memory(image_bytes_io)
        task_ctx.image_bytes = image_bytes_io.getvalue(); task_ctx.image_filename = f"webhook_tg_user_{user_id}_msg_{message_id}"
        try:
            pil_image = Image.open(io.BytesIO(task_ctx.image_bytes)); image_format = pil_image.format or "JPEG"
            mime_type_map = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "BMP": "image/bmp", "WEBP": "image/webp"}
            task_ctx.image_mime_type = mime_type_map.get(image_format.upper(), "image/jpeg")
            task_ctx.image_filename += f".{image_format.lower()}"
            logger.info(f"handle_photo: Imagen descargada. Formato: {image_format}, MIME: {task_ctx.image_mime_type}")
        except Exception as e_pil: logger.error(f"handle_photo: Error Pillow: {e_pil}. Usando image/jpeg.", exc_info=True); task_ctx.image_mime_type = "image/jpeg"; task_ctx.image_filename += ".jpg"
        
        updated_task_ctx = await orchestrate_image_processing(task_ctx)
        analysis_result = updated_task_ctx.equation_analysis_db

        if analysis_result and analysis_result.latex:
            message_parts = ["✅ <b>Análisis (Webhook v0.10.1) Completado!</b>"]
            if analysis_result.database_status == "Retrieved from DB": message_parts.append("<i>(Info de DB)</i>")
            def add_field(lbl, val, code=False):
                if val and str(val).strip(): esc_val = html.escape(str(val).strip()); message_parts.append(f"\n<b>{html.escape(lbl)}:</b>{('<code>'+esc_val+'</code>') if code else (' '+esc_val)}")
            add_field("Nombre", analysis_result.name); add_field("Categoría(s)", analysis_result.category)
            add_field("Descripción", analysis_result.description); add_field("LaTeX (Extraído)", analysis_result.latex, True)
            add_field("Derivación", analysis_result.derivation, True); add_field("Usos Comunes", analysis_result.uses)
            add_field("Variables", analysis_result.vars); add_field("Ecuaciones Similares", analysis_result.similars)
            message_parts.append(f"\n\n<tg-spoiler><i>Debug: LLM:{html.escape(analysis_result.llm_analysis_status)},DB:{html.escape(analysis_result.database_status)},ID:{analysis_result.equation_id}</i></tg-spoiler>")
            final_text = "\n".join(message_parts);
            if len(final_text) > 4096: final_text = final_text[:4090] + "\n(...)"
            await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text=final_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            if analysis_result.latex: 
                img_buf = render_latex_to_image_bytes(analysis_result.latex, f"eq_{analysis_result.equation_id}.png")
                if img_buf:
                    try: img_buf.seek(0); await context.bot.send_photo(chat_id=chat_id, photo=img_buf, caption="Ecuación Renderizada", reply_to_message_id=processing_msg.message_id) 
                    finally:
                        if img_buf: img_buf.close()
        elif analysis_result and updated_task_ctx.error_message: await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text=f"😕 Error: {html.escape(updated_task_ctx.error_message)}")
        elif analysis_result and not analysis_result.latex: await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text=f"😕 No pude extraer LaTeX. Estado: {html.escape(analysis_result.llm_analysis_status)}")
        else: await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text="Error procesando.")
    except Exception as e_photo_main:
        logger.error(f"handle_photo: Error crítico: {e_photo_main}", exc_info=True)
        try: await context.bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text="🤕 ¡Ups! Algo salió muy mal procesando la foto.")
        except: pass

async def error_handler(update: object | None, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"PTB Exception en error_handler: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try: await context.bot.send_message(chat_id=update.effective_chat.id, text="Error inesperado al procesar tu solicitud.")
        except Exception as e_err_handler: logger.error(f"Error en error_handler al notificar: {e_err_handler}")

# --- Lógica del Webhook y Ciclo de Vida de FastAPI ---
async def setup_ptb_app_for_webhook():
    global ptb_application 
    if not TELEGRAM_BOT_TOKEN:
        logger.error("setup_ptb_app_for_webhook: TELEGRAM_BOT_TOKEN no configurado.")
        return None
    logger.info("setup_ptb_app_for_webhook: Configurando Application de PTB...")
    try:
        ptb_request = telegram.request.HTTPXRequest(connect_timeout=5.0, read_timeout=10.0)
        ptb_application_builder = Application.builder().token(TELEGRAM_BOT_TOKEN).request(ptb_request)
        ptb_application = ptb_application_builder.build()
        logger.info(f"setup_ptb_app_for_webhook: Instancia de Application PTB creada con ID: {id(ptb_application)}")
        ptb_application.add_handler(TypeHandler(Update, generic_update_handler), group=-1)
        logger.info("setup_ptb_app_for_webhook: Handler 'generic_update_handler' añadido.")
        ptb_application.add_handler(CommandHandler("start", start_command))
        logger.info("setup_ptb_app_for_webhook: Handler 'start_command' añadido.")
        ptb_application.add_handler(CommandHandler("help", help_command))
        logger.info("setup_ptb_app_for_webhook: Handler 'help_command' añadido.")
        ptb_application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        logger.info("setup_ptb_app_for_webhook: Handler 'handle_photo' añadido.")
        ptb_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
        logger.info("setup_ptb_app_for_webhook: Handler 'handle_text_message' añadido.")
        ptb_application.add_error_handler(error_handler)
        logger.info("setup_ptb_app_for_webhook: Handler 'error_handler' añadido.")
        logger.info("setup_ptb_app_for_webhook: Aplicación de Telegram configurada con handlers.")
        return ptb_application
    except Exception as e:
        logger.error(f"setup_ptb_app_for_webhook: Error crítico durante la inicialización de PTB: {e}", exc_info=True)
        ptb_application = None
        return None

@asynccontextmanager
async def lifespan(fastapi_app_instance: FastAPI):
    global ptb_application, db_client, db_collection
    logger.info("LIFESPAN FastAPI [STARTUP]: Iniciando...")
    if MONGO_CONNECTION_STRING:
        try:
            logger.info("LIFESPAN Startup: Conectando a MongoDB/CosmosDB...")
            db_client = pymongo.MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=10000)
            await asyncio.to_thread(db_client.admin.command, 'ping')
            db = db_client[DATABASE_NAME] 
            db_collection = db[COLLECTION_NAME] 
            logger.info(f"LIFESPAN Startup: Conexión a DB exitosa. DB: '{DATABASE_NAME}', Colección: '{COLLECTION_NAME}'")
        except Exception as e_db:
            logger.error(f"LIFESPAN Startup: Error al conectar/configurar MongoDB/CosmosDB: {e_db}", exc_info=True)
            db_client = None; db_collection = None
    else:
        logger.warning("LIFESPAN Startup: COSMOS_MONGO_CONNECTION_STRING no configurada. DB desactivada.")

    await setup_ptb_app_for_webhook()

    if ptb_application and WEBHOOK_URL_BASE and TELEGRAM_BOT_TOKEN:
        full_webhook_url = f"{WEBHOOK_URL_BASE.rstrip('/')}/{TELEGRAM_BOT_TOKEN.split(':')[0]}_{WEBHOOK_SECRET_PATH_SUFFIX}"
        logger.info(f"LIFESPAN Startup: Intentando configurar webhook en: {full_webhook_url}")
        try:
            await ptb_application.initialize() 
            logger.info(f"LIFESPAN Startup: ptb_application inicializado. Bot ID: {ptb_application.bot.id}")
            current_webhook_info = await ptb_application.bot.get_webhook_info()
            logger.info(f"LIFESPAN Startup: Webhook actual: URL='{current_webhook_info.url}', pending_update_count={current_webhook_info.pending_update_count}")
            if current_webhook_info.url != full_webhook_url:
                logger.info(f"LIFESPAN Startup: El webhook actual ('{current_webhook_info.url}') no coincide. Configurando nuevo webhook...")
                success = await ptb_application.bot.set_webhook(
                    url=full_webhook_url,
                    allowed_updates=Update.ALL_TYPES, 
                    drop_pending_updates=True, 
                )
                if success: logger.info(f"LIFESPAN Startup: Webhook configurado exitosamente en {full_webhook_url}")
                else: logger.error("LIFESPAN Startup: Fallo al configurar el webhook (set_webhook devolvió False).")
            else: logger.info("LIFESPAN Startup: El webhook ya está configurado correctamente a la URL esperada.")
        except Exception as e_webhook_setup:
            logger.error(f"LIFESPAN Startup: Error inesperado al configurar el webhook: {e_webhook_setup}", exc_info=True)
    elif not ptb_application: logger.warning("LIFESPAN Startup: ptb_application no configurada, no se puede establecer webhook.")
    elif not WEBHOOK_URL_BASE: logger.warning("LIFESPAN Startup: WEBHOOK_URL_BASE no configurada, no se establecerá webhook.")

    logger.info("LIFESPAN FastAPI [STARTUP]: Procesos de fondo y webhook (intentado) configurados.")
    yield 
    
    logger.info("LIFESPAN FastAPI [SHUTDOWN]: Deteniendo procesos de fondo...")
    if ptb_application and hasattr(ptb_application, 'bot') and os.getenv("DELETE_WEBHOOK_ON_SHUTDOWN", "true").lower() == "true":
        logger.info("LIFESPAN Shutdown: Intentando eliminar el webhook de Telegram...")
        try:
            success = await ptb_application.bot.delete_webhook(drop_pending_updates=True)
            if success: logger.info("LIFESPAN Shutdown: Webhook eliminado exitosamente.")
            else: logger.warning("LIFESPAN Shutdown: Fallo al eliminar el webhook (delete_webhook devolvió False).")
        except Exception as e_webhook_delete:
            logger.error(f"LIFESPAN Shutdown: Error inesperado al eliminar el webhook: {e_webhook_delete}", exc_info=True)

    if ptb_application and hasattr(ptb_application, 'shutdown') and callable(ptb_application.shutdown):
        logger.info("LIFESPAN Shutdown: Llamando a ptb_application.shutdown()...")
        try:
            await ptb_application.shutdown()
            logger.info("LIFESPAN Shutdown: ptb_application.shutdown() completado.")
        except Exception as e_ptb_shutdown:
            logger.error(f"LIFESPAN Shutdown: Error durante ptb_application.shutdown(): {e_ptb_shutdown}", exc_info=True)
    
    if db_client:
        logger.info("LIFESPAN Shutdown: Cerrando conexión con MongoDB/CosmosDB...")
        try:
            db_client.close()
            logger.info("LIFESPAN Shutdown: Conexión con MongoDB/CosmosDB cerrada.")
        except Exception as e_db_close:
            logger.error(f"LIFESPAN Shutdown: Error al cerrar la conexión con MongoDB: {e_db_close}", exc_info=True)
    logger.info("LIFESPAN FastAPI [SHUTDOWN]: Limpieza de procesos de fondo completada.")

app.router.lifespan_context = lifespan

WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN.split(':')[0]}_{WEBHOOK_SECRET_PATH_SUFFIX}" if TELEGRAM_BOT_TOKEN else f"/telegram_webhook_placeholder_{WEBHOOK_SECRET_PATH_SUFFIX}"

# MODIFICACIÓN: Endpoint del Webhook Simplificado para Debugging
# ... (importaciones al principio)
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks # Asegúrate de que BackgroundTasks esté importado
# ...

# Modifica tu endpoint del webhook:
@app.post(WEBHOOK_PATH)
async def telegram_webhook_endpoint(request: Request, background_tasks: BackgroundTasks): # Añadir background_tasks
    global ptb_application
    # ... (código para obtener update_data y deserializar el Update como antes) ...
    if not ptb_application: # Copiado de tu v0.10.1
        logger.error(f"Webhook ({WEBHOOK_PATH}): ptb_application no está inicializada...")
        return Response(content="Error interno: Bot no inicializado.", status_code=500)
    try:
        update_data = await request.json()
        logger.info(f"Webhook ({WEBHOOK_PATH}): JSON bruto recibido: {json.dumps(update_data, indent=2, ensure_ascii=False)}")
    except json.JSONDecodeError:
        logger.error(f"Webhook ({WEBHOOK_PATH}): Error decodificando JSON del request.")
        return Response(content="Cuerpo del request no es JSON válido.", status_code=400)
    except Exception as e_req_body:
        logger.error(f"Webhook ({WEBHOOK_PATH}): Error obteniendo cuerpo del request: {e_req_body}", exc_info=True)
        return Response(content="Error leyendo el request.", status_code=500)

    try:
        if not ptb_application.bot:
            logger.error(f"Webhook ({WEBHOOK_PATH}): ptb_application.bot no está disponible.")
            return Response(content="Error interno: Bot no completamente inicializado para deserializar.", status_code=500)

        update = Update.de_json(update_data, ptb_application.bot)
        logger.info(f"Webhook ({WEBHOOK_PATH}): Update ID {update.update_id} deserializado correctamente.")

        # --- USAR BACKGROUND TASK ---
        background_tasks.add_task(ptb_application.process_update, update)
        logger.info(f"Webhook ({WEBHOOK_PATH}): Update ID {update.update_id} añadido a background_tasks para procesamiento por PTB.")
        # ---------------------------
        
        return Response(status_code=200, content="Update recibido y programado para procesamiento.") # Respuesta rápida

    except Exception as e_process_update:
        logger.error(f"Webhook ({WEBHOOK_PATH}): Error al deserializar Update o programar la tarea: {e_process_update}", exc_info=True)
        return Response(content="Error interno procesando el update con PTB.", status_code=500)
            
@app.get("/", summary="Endpoint raíz de la API de Raphael (Webhook Mode)")
async def root_endpoint():
    # MODIFICACIÓN AQUÍ:
    db_status = "Conectado" if db_collection is not None else "Desconectado" # Comparar con None

    webhook_url_telegram_should_have = "No configurado"
    if WEBHOOK_URL_BASE and TELEGRAM_BOT_TOKEN:
        webhook_url_telegram_should_have = f"{WEBHOOK_URL_BASE.rstrip('/')}/{TELEGRAM_BOT_TOKEN.split(':')[0]}_{WEBHOOK_SECRET_PATH_SUFFIX}"

    return {
        "message": f"Raphael API (v{app.version}) en modo Webhook está operativa.",
        "webhook_base_url_env": WEBHOOK_URL_BASE or "No configurado en .env",
        "expected_webhook_url_for_telegram": webhook_url_telegram_should_have,
        "database_status": db_status
    }

if __name__ == "__main__":
    logger.info(f"Ejecutando script '{__file__}' directamente...")
    logger.info("Este modo es para desarrollo local con Uvicorn. El webhook debe configurarse manualmente con ngrok o similar.")
    logger.info("Asegúrate de que la variable de entorno WEBHOOK_URL_BASE esté configurada a tu URL de ngrok (HTTPS).")
    
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True, 
        log_level="info" # Uvicorn log level, no el de la app. El de la app se configura con logging.basicConfig
    )