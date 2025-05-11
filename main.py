# -*- coding: utf-8 -*-
# Versión 0.5.3.14 - Eliminado updater.idle() de la tarea de polling
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File
import asyncio 
import requests 
import os
import uuid
from pydantic import BaseModel, ValidationError
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

# --- Google Gemini Imports ---
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.INFO) 
logger = logging.getLogger(__name__)

# --- Crear Instancia FastAPI ---
app = FastAPI(title="Raphael Agent API & Telegram Bot", version="0.5.3.14")

# --- Variables de Entorno ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_CONNECTION_STRING = os.getenv("COSMOS_MONGO_CONNECTION_STRING")
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "raphaeldb")
COLLECTION_NAME = os.getenv("COSMOS_COLLECTION_NAME", "equations")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


logger.info(f"GOOGLE_API_KEY loaded: {'Yes' if GOOGLE_API_KEY else 'NO'}")
logger.info(f"COSMOS_MONGO_CONNECTION_STRING loaded: {'Yes' if MONGO_CONNECTION_STRING else 'NO'}")
logger.info(f"TELEGRAM_BOT_TOKEN loaded: {'Yes' if TELEGRAM_BOT_TOKEN else 'NO'}")

# --- Configurar Google Gemini ---
genai_configured = False 
gemini_model_multimodal = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model_multimodal = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Google AI SDK configurado y modelo 'gemini-2.0-flash-latest' cargado.")
        genai_configured = True
    except Exception as e:
        logger.error(f"Error config Google AI SDK: {e}", exc_info=True)
else:
    logger.warning("GOOGLE_API_KEY no configurada para Gemini.")

# --- Configuración Base de Datos ---
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

# --- Lista Permitida de Categorías ---
ALLOWED_CATEGORIES = [
    "mecánica clasica", "mecánica cuantica", "algebra lineal", "cálculo",
    "ecuaciones diferenciales parciales", "ecuaciones diferenciales ordinarias",
    "estadística y probabilidad", "termodinámica", "relatividad",
    "series (taylor, fourier, laurent, etc...)", "genericas", "óptica", "electromagnetismo"
]

# --- Modelos de Datos ---
class LLMAnalysisData(BaseModel):
    latex_extracted_from_image: str | None = None
    name: str | None = None
    category: str | None = None
    description: str | None = None
    derivation: str | None = None
    uses: str | None = None
    vars: str | None = None
    similars: str | None = None

class EquationAnalysis(BaseModel):
    equation_id: str
    latex: str
    name: str | None = None
    category: str | None = None
    description: str | None = None
    derivation: str | None = None
    uses: str | None = None
    vars: str | None = None
    similars: str | None = None
    llm_analysis_status: str
    database_status: str

# --- Función de Normalización de LaTeX ---
def normalize_latex(latex_str: str) -> str | None:
    if not latex_str: return None
    normalized = re.sub(r'\s+', ' ', latex_str).strip()
    if normalized.startswith("$$") and normalized.endswith("$$"): normalized = normalized[2:-2].strip()
    if normalized.startswith("\\[") and normalized.endswith("\\]"): normalized = normalized[2:-2].strip()
    if normalized.startswith("$") and normalized.endswith("$"): normalized = normalized[1:-1].strip()
    if sympy_available and parse_latex and sympy_latex:
        try:
            sympy_expr = parse_latex(normalized)
            normalized_sympy = sympy_latex(sympy_expr, mode='inline', mul_symbol='dot')
            return normalized_sympy.strip()
        except Exception: return normalized
    return normalized

# --- Función de Búsqueda en DB ---
async def find_equation_by_normalized_latex(normalized_latex_key: str) -> dict | None:
    if db_collection is None or not normalized_latex_key: return None
    try:
        return db_collection.find_one({"normalized_latex_key": normalized_latex_key})
    except Exception as e: logger.error(f"Error buscando en DB: {e}", exc_info=True); return None

# --- Función para Llamar a Gemini CON IMAGEN (OCR + Análisis) ---
async def extract_and_analyze_equation_from_image_with_gemini(image_bytes: bytes, image_mime_type: str = "image/jpeg") -> LLMAnalysisData | None:
    if not genai_configured or not gemini_model_multimodal:
        logger.error("Gemini (multimodal) no configurado/cargado.")
        return None
    if not image_bytes: 
        logger.error("Bytes de imagen vacíos para Gemini.")
        return None
        
    try:
        category_list_str = ", ".join([f'"{cat}"' for cat in ALLOWED_CATEGORIES])
        image_part_for_gemini = {"mime_type": image_mime_type, "data": image_bytes}

        prompt_parts = [
            "Tarea: Extraer la ecuación matemática principal de la imagen proporcionada y luego analizarla detalladamente.",
            "Paso 1: Observa la imagen. Identifica la ecuación matemática más prominente o central. Extrae esta ecuación y preséntala en formato LaTeX.",
            "Paso 2: Usando ÚNICAMENTE el LaTeX que extrajiste en el Paso 1, analiza la ecuación.",
            f"""
Devuelve tu análisis completo (incluyendo el LaTeX extraído) estrictamente en formato JSON y SIEMPRE EN ESPAÑOL. El JSON debe tener la siguiente estructura y campos:
{{
  "latex_extracted_from_image": "El LaTeX que extrajiste de la imagen en el Paso 1. Asegúrate de que sea LaTeX válido y completo.",
  "name": "Un nombre común o descriptivo para la ecuación/fórmula extraída. Si no tiene uno específico, intenta inferir uno.",
  "category": "Clasifica la ecuación en una o más de las siguientes categorías: [{category_list_str}]. Si aplica a múltiples categorías, sepáralas con ' / '. Si ninguna categoría específica de la lista aplica claramente, usa 'genericas'.",
  "description": "Una breve descripción de lo que representa la ecuación, su propósito principal o el principio físico/matemático que encapsula.",
  "derivation": "PROPORCIONA OBLIGATORIAMENTE UNA DERIVACIÓN MATEMÁTICA ANALÍTICA paso a paso de la ecuación dada, en formato LaTeX. \
Prioridad 1: Presenta la derivación GENERAL completa o los pasos fundamentales de la misma. No te preocupes por la longitud o la complejidad percibida; asume que el usuario tiene el conocimiento para entenderla. Muestra todos los pasos matemáticos relevantes. \
Prioridad 2 (SOLO si la derivación general es genuinamente intratable en este formato): Presenta la derivación para un CASO PARTICULAR REPRESENTATIVO O UNA VERSIÓN SIMPLIFICADA de la ecuación. Detalla los pasos matemáticos para este caso. Por ejemplo, para la ecuación de energía del átomo de Hidrógeno, deriva el nivel n=1 o n=2 explícitamente desde la ecuación de Schrödinger radial simplificada, mostrando los pasos. \
Prioridad 3 (ÚLTIMO RECURSO, si es un postulado/definición): Indica 'Postulado Fundamental: [Explicación concisa de su origen o contexto]' o 'Definición Axiomática: [Explicación concisa]'. \
BAJO NINGUNA CIRCUNSTANCIA respondas con frases como 'Derivación compleja omitida', 'requiere conocimiento profundo', 'no es posible presentarla aquí' o justificaciones sobre la dificultad. El objetivo es SIEMPRE mostrar un procedimiento analítico o el fundamento de la ecuación. Utiliza LaTeX válido y claro.",
  "uses": "Menciona algunos usos comunes, aplicaciones prácticas o áreas donde esta ecuación es frecuentemente empleada. Sé conciso.",
  "vars": "Define cada una de las variables presentes en la ecuación LaTeX extraída. Formatea esto como una cadena de texto donde cada par variable-definición esté claramente separado (ej., 'E: Energía total, m: masa, c: velocidad de la luz').
  "similars": "Enumera otras ecuaciones o fórmulas que sean similares o relacionadas con la ecuación extraída. Puedes incluir ecuaciones que sean variaciones, extensiones o aplicaciones de la ecuación original. Sé conciso."
}}

Consideraciones Importantes:
- El campo "latex_extracted_from_image" es crucial. Si no puedes extraer un LaTeX claro de la imagen, indica "No se pudo extraer LaTeX de la imagen" en este campo y deja los demás campos de análisis como null o "No aplicable".
- Asegúrate de que todos los valores en el JSON sean cadenas de texto.
- Intenta completar todos los campos de análisis basados en el LaTeX extraído.
- El formato LaTeX debe ser compatible para renderizar con Matplotlib.
""",
            image_part_for_gemini
        ]

        logger.info(f"Enviando prompt multimodal (v0.5.3.7) a Gemini...")
        generation_config = GenerationConfig(temperature=0.3, response_mime_type="application/json") 
        response = await gemini_model_multimodal.generate_content_async(prompt_parts, generation_config=generation_config)

        if response.parts:
            response_text = response.text.strip()
            logger.info(f"Respuesta CRUDA Gemini (JSON esperado): {response_text[:500]}...")

            llm_json_data = None
            try:
                llm_json_data = json.loads(response_text)
                extracted_latex_from_llm = llm_json_data.get('latex_extracted_from_image')
                if not extracted_latex_from_llm or "no se pudo extraer" in extracted_latex_from_llm.lower():
                    logger.warning(f"Gemini no pudo extraer LaTeX: '{extracted_latex_from_llm}'")
                    return None 
                
                category_str = llm_json_data.get('category', '')
                validated_categories_list = []
                if isinstance(category_str, str) and category_str.strip():
                    potential_categories = [cat.strip().lower() for cat in category_str.split('/')]
                    for cat_item in potential_categories:
                        if cat_item in ALLOWED_CATEGORIES: validated_categories_list.append(cat_item)
                llm_json_data['category'] = " / ".join(validated_categories_list) if validated_categories_list else 'genericas'

                for field_name in LLMAnalysisData.model_fields.keys():
                    if field_name not in llm_json_data: llm_json_data[field_name] = None
                    elif isinstance(llm_json_data[field_name], str) and not llm_json_data[field_name].strip() and field_name not in ['latex_extracted_from_image', 'category']:
                        llm_json_data[field_name] = None

                return LLMAnalysisData(**llm_json_data)
            except json.JSONDecodeError as e: logger.error(f"Error JSON Gemini: {e}. Respuesta: {response_text}"); return None
            except ValidationError as e: logger.error(f"Error Pydantic Gemini: {e}. Datos: {llm_json_data}"); return None
            except Exception as e: logger.error(f"Error procesando resp Gemini: {e}", exc_info=True); return None
        else:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 logger.warning(f"Respuesta bloqueada. Razón: {response.prompt_feedback.block_reason}.")
            else: logger.warning(f"Respuesta Gemini vacía/sin parts. Respuesta: {response}")
            return None
    except Exception as e: logger.error(f"Error API Gemini: {e}", exc_info=True); return None

# --- Función para Guardar en MongoDB/Cosmos ---
async def save_analysis_to_mongo(analysis_data: EquationAnalysis, normalized_latex_key: str) -> bool:
    if db_collection is None: logger.warning("DB no configurada."); return False
    try:
        doc = analysis_data.dict(); doc['_id'] = analysis_data.equation_id; doc['normalized_latex_key'] = normalized_latex_key
        res = db_collection.replace_one({'_id': analysis_data.equation_id}, doc, upsert=True)
        return res.acknowledged
    except Exception as e: logger.error(f"Error guardando DB: {e}", exc_info=True); return False

# --- Función para Renderizar LaTeX a Imagen ---
def render_latex_to_image_bytes(latex_str: str | None, filename: str = "equation.png") -> io.BytesIO | None:
    if not latex_str or not latex_str.strip(): 
        logger.warning(f"Render LaTeX inválido/vacío: {filename}")
        return None
    txt = latex_str.strip()
    if not (txt.startswith('$') and txt.endswith('$')) and not txt.startswith('\\['): 
        txt = f"${txt}$"
    fig = None 
    try:
        num_lines = len(txt.splitlines())
        fig_height = max(2.5, num_lines * 0.6 + 1.0) 
        fig_width = 10 
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
        fig.patch.set_facecolor('white')
        ax.text(0.5, 0.5, txt, fontsize=18, ha='center', va='center', wrap=True) # Centrado
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

# --- Lógica Principal de Procesamiento ---
async def process_equation(image_bytes: bytes, image_mime_type: str, filename: str = "telegram_image") -> EquationAnalysis | None:
    llm_data = await extract_and_analyze_equation_from_image_with_gemini(image_bytes, image_mime_type)
    if not llm_data or not llm_data.latex_extracted_from_image:
        logger.error("Fallo OCR/Análisis LLM inicial."); 
        return None
    
    extracted_latex = llm_data.latex_extracted_from_image
    normalized_key = normalize_latex(extracted_latex) or extracted_latex

    if db_collection is not None and (existing_doc := await find_equation_by_normalized_latex(normalized_key)):
        try: 
            return EquationAnalysis(**existing_doc, database_status="Retrieved from DB", llm_analysis_status=existing_doc.get('llm_analysis_status', 'Retrieved'))
        except ValidationError as e: 
            logger.error(f"Error validar datos DB: {e}. Re-analizando.")

    analysis = EquationAnalysis(
        equation_id=str(uuid.uuid4()), latex=extracted_latex,
        name=llm_data.name, category=llm_data.category, description=llm_data.description,
        derivation=llm_data.derivation, uses=llm_data.uses, vars=llm_data.vars,
        llm_analysis_status="Success (OCR & Analysis)" if llm_data.latex_extracted_from_image else "Failed (LLM error or no data)",
        database_status="Save Attempt Pending"
    )
    
    if db_collection is not None and await save_analysis_to_mongo(analysis, normalized_key):
        analysis.database_status = "Saved Successfully (New Entry)"
    elif db_collection is None:
        analysis.database_status = "Skipped (DB Unavailable)"
    else:
        analysis.database_status = "Save Failed"
        
    return analysis

# --- Manejadores Telegram ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(f"¡Hola {update.effective_user.mention_html()}! Envíame foto de ecuación.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html("Envíame foto de ecuación. <b>Consejos:</b> Imagen clara.")
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Por favor, envíame una foto de una ecuación.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo: return
    chat_id = update.effective_chat.id
    msg_edit = await context.bot.send_message(chat_id=chat_id, text="📸 Procesando...")
    try:
        photo_f = await update.message.photo[-1].get_file()
        img_bytes_io = io.BytesIO(); await photo_f.download_to_memory(img_bytes_io)
        img_bytes = img_bytes_io.getvalue()
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_fmt = img_pil.format or "JPEG"; img_mime = Image.MIME.get(img_fmt.upper(), "image/jpeg")
        
        res = await process_equation(img_bytes, img_mime, f"tg_{photo_f.file_id}.{img_fmt.lower()}")
        if res:
            parts = ["✅ <b>¡Análisis Completado!</b>"]
            if res.database_status == "Retrieved from DB": parts.append("<i>(Info de DB)</i>")
            def add(lbl, val, code=False):
                if val and str(val).strip(): parts.append(f"\n<b>{html.escape(lbl)}:</b>{'<code>'+html.escape(str(val).strip())+'</code>' if code else ' '+html.escape(str(val).strip())}")
            add("Nombre", res.name); add("Categoría(s)", res.category); add("Descripción", res.description)
            add("Usos", res.uses); add("LaTeX (Texto)", res.latex, True); add("Derivación", res.derivation, True); add("Variables", res.vars, True)
            parts.append(f"\n<tg-spoiler><i>Debug: LLM:{html.escape(res.llm_analysis_status)},DB:{html.escape(res.database_status)}</i></tg-spoiler>")
            txt = "\n".join(parts); txt = txt[:4090] + "\n(...)" if len(txt) > 4096 else txt
            await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_edit.message_id, text=txt, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            
            if res.latex and (img_buf := render_latex_to_image_bytes(res.latex, f"eq_{res.equation_id}.png")):
                try:
                    img_buf.seek(0)
                    await context.bot.send_photo(chat_id=chat_id, photo=img_buf, caption="Ecuación (Renderizada)", reply_to_message_id=msg_edit.message_id)
                    logger.info(f"Imagen renderizada para LaTeX: {res.latex[:30]}... enviada.") 
                except Exception as e: logger.error(f"Error enviando imagen renderizada: {e}", exc_info=True)
                finally: 
                    if img_buf: img_buf.close()
            else: logger.warning(f"No se renderizó/envió imagen para LaTeX: {res.latex[:50] if res.latex else 'N/A'}")
        else: await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_edit.message_id, text="Error procesando ecuación.")
    except Exception as e:
        logger.error(f"Error en handle_photo: {e}", exc_info=True)
        try: await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_edit.message_id, text="Ups, error. 🤕")
        except: await context.bot.send_message(chat_id=chat_id, text="Ups, error. 🤕")

async def error_handler(update: object | None, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Excepción no controlada: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try: await context.bot.send_message(chat_id=update.effective_chat.id, text="Error inesperado.")
        except: pass

# --- Configuración App Bot Telegram y FastAPI Lifespan ---
telegram_app: Application | None = None
polling_task: asyncio.Task | None = None 

if TELEGRAM_BOT_TOKEN:
    try:
        telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        telegram_app.add_handler(CommandHandler("start", start_command)); telegram_app.add_handler(CommandHandler("help", help_command))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)); telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        telegram_app.add_error_handler(error_handler)
        logger.info("App Telegram configurada.")
    except Exception as e: 
        logger.error(f"Error init Telegram: {e}", exc_info=True)
        telegram_app = None 

async def run_telegram_polling():
    logger.info("Entrando a run_telegram_polling()...")
    if telegram_app and telegram_app.updater:
        try:
            logger.info("Llamando a telegram_app.initialize() antes de start_polling...")
            await telegram_app.initialize() # <--- LLAMADA EXPLÍCITA A INITIALIZE
            logger.info("telegram_app.initialize() completado. Iniciando polling...")
            
            await telegram_app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
            logger.info("Polling de Telegram iniciado y corriendo (después de start_polling).")
            
            # En lugar de updater.idle(), que no existe, podemos simplemente mantener la tarea viva
            # mientras el updater esté corriendo. La cancelación vendrá del lifespan.
            while telegram_app.updater.running:
                await asyncio.sleep(1) # Chequea cada segundo, o un valor mayor
            logger.info("El bucle de polling (while updater.running) ha terminado.")

        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            logger.info("Polling de Telegram interrumpido o cancelado en run_telegram_polling.")
        except Exception as e: 
            logger.error(f"Error EXCEPCIONAL en la ejecución del polling de Telegram: {e}", exc_info=True)
        finally:
            logger.info("Bloque finally de run_telegram_polling alcanzado.")
            if telegram_app and telegram_app.updater and telegram_app.updater.running:
                logger.info("Deteniendo updater de polling de Telegram (finally de run_telegram_polling).")
                await telegram_app.updater.stop()
            if telegram_app: 
                logger.info("Ejecutando shutdown de la aplicación Telegram (finally de run_telegram_polling).")
                await telegram_app.shutdown()
            logger.info("Tarea de polling run_telegram_polling finalizada completamente.")
    else:
        logger.error("telegram_app o telegram_app.updater no están disponibles en run_telegram_polling.")


from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app_lifespan: FastAPI):
    global polling_task
    logger.info("FastAPI lifespan STARTUP: Iniciando tareas de fondo...")
    if telegram_app and telegram_app.updater: 
        polling_task = asyncio.create_task(run_telegram_polling())
        logger.info(f"Tarea de polling de Telegram creada. Estado inicial de la tarea: {polling_task}")
        await asyncio.sleep(2) 
        if polling_task.done():
            try:
                exc = polling_task.exception()
                if exc:
                     logger.error(f"La tarea de polling falló inmediatamente después de crearla: {exc}", exc_info=exc)
            except asyncio.CancelledError:
                 logger.info("La tarea de polling fue cancelada inmediatamente (raro).")
            except asyncio.InvalidStateError:
                 logger.info("La tarea de polling está en un estado inválido (probablemente cancelada y ya se obtuvo la excepción).")

    elif not telegram_app:
        logger.error("Aplicación Telegram no inicializada. No se puede iniciar polling en lifespan de FastAPI.")
    elif not telegram_app.updater:
        logger.error("Telegram updater no disponible. No se puede iniciar polling.")
    yield
    logger.info("FastAPI lifespan SHUTDOWN: Deteniendo tareas de fondo...")
    if polling_task and not polling_task.done():
        logger.info("Cancelando tarea de polling de Telegram...")
        polling_task.cancel()
        try:
            await polling_task 
        except asyncio.CancelledError:
            logger.info("Tarea de polling de Telegram cancelada exitosamente durante el shutdown.")
        except Exception as e_task_shutdown:
            logger.error(f"Error esperando la cancelación de la tarea de polling: {e_task_shutdown}")
    
    if telegram_app and telegram_app.updater and telegram_app.updater.running:
        logger.warning("Updater de Telegram seguía corriendo durante el shutdown del lifespan. Intentando detener de nuevo.")
        await telegram_app.updater.stop()
    if telegram_app: 
        logger.warning("Aplicación PTB podría seguir activa durante el shutdown del lifespan. Intentando shutdown de nuevo.")
        await telegram_app.shutdown()
    logger.info("Lifespan de FastAPI finalizado.")

app.router.lifespan_context = lifespan

# --- Endpoint Webhook y Raíz ---
TELEGRAM_WEBHOOK_SECRET_PATH = f"/{TELEGRAM_BOT_TOKEN.split(':')[0]}_wh" if TELEGRAM_BOT_TOKEN else "/tg_wh_raphael_fb"
@app.post(TELEGRAM_WEBHOOK_SECRET_PATH)
async def telegram_webhook(req: Request):
    if not telegram_app: return Response(status_code=503)
    try: upd = Update.de_json(await req.json(), telegram_app.bot); await telegram_app.process_update(upd); return Response(status_code=200)
    except Exception as e: logger.error(f"Error webhook: {e}", exc_info=True); return Response(status_code=500)

@app.get("/")
async def root(): return {"message": f"Raphael API (v{app.version}) funcionando."}

# --- Ejecución Directa (Polling Bloqueante) ---
if __name__ == "__main__":
    if telegram_app:
        logger.info("Ejecución directa: Iniciando polling bloqueante...")
        try: telegram_app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        except KeyboardInterrupt: logger.info("Polling directo detenido.")
        except Exception as e: logger.error(f"Error crítico polling directo: {e}", exc_info=True)
    else: logger.error("Error Crítico: Telegram app no init. Polling directo no iniciado.")