import os
import re
import json
import hmac
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading

import requests
from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv

# Load local environment if available
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dairy_mate_whatsapp")

# Environment configurations
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dairy_mate")
VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN", "")
APP_SECRET = os.environ.get("META_APP_SECRET", "")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
GRAPH_API_VERSION = os.environ.get("WHATSAPP_GRAPH_API_VERSION", "v21.0")
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# MongoDB client initialization
if not MONGODB_URI:
    logger.warning("MONGODB_URI is not set. Database persistence will be disabled.")
    db = None
    messages_collection = None
    sessions_collection = None
else:
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[MONGODB_DB_NAME]
        messages_collection = db["whatsapp_messages"]
        sessions_collection = db["user_sessions"]
        logger.info("Connected to MongoDB database: %s", MONGODB_DB_NAME)
    except Exception as e:
        logger.error("Failed to connect to MongoDB: %s", e)
        db = None
        messages_collection = None
        sessions_collection = None

# Async executor for non-blocking webhook processing
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="dm_worker")

# Model API
from model_api import get_screener  # noqa: E402

app = Flask(__name__)


def startup_background_tasks():
    """Pre-warm ONNX models and recover any recently interrupted jobs."""
    logger.info("Starting background pre-warming of AI models...")
    for kind in ("udder", "teat", "tick"):
        try:
            get_screener(kind)
            logger.info("Pre-warmed model: %s", kind)
        except Exception as e:
            logger.warning("Could not pre-warm model %s: %s", kind, e)

    # Recovery of uncompleted tasks from recent worker restarts
    if messages_collection is not None:
        try:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
            interrupted = list(messages_collection.find({
                "status": "received",
                "received_at": {"$gte": cutoff}
            }).limit(20))
            if interrupted:
                logger.info("Recovering %d recent unprocessed messages after startup", len(interrupted))
                for item in interrupted:
                    raw = item.get("raw_message")
                    if raw:
                        executor.submit(process_message, raw)
        except Exception as err:
            logger.warning("Startup message recovery check encountered an error: %s", err)


# Start pre-warming and recovery in background daemon thread
threading.Thread(target=startup_background_tasks, daemon=True).start()


def parse_model_kind(text):
    """Detect intended check from caption or message text."""
    if not text:
        return None
    cleaned = text.strip().lower()
    match = re.search(r"\b(udder|teat|tick)\b", cleaned)
    if match:
        return match.group(1)
    return None


def send_text(to, body):
    """Send a WhatsApp text message via Cloud API."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        logger.error("Cannot send WhatsApp reply: Cloud API credentials not configured.")
        return False
    try:
        url = f"{GRAPH_BASE}/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "DairyMateAI/1.0"
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": body[:4000]},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if not response.ok:
            error_data = {}
            try:
                error_data = response.json().get("error", {})
            except Exception:
                pass
            err_code = error_data.get("code")
            err_msg = error_data.get("message", response.text[:300])

            if err_code == 190:
                logger.error("WhatsApp Cloud API EXPIRED TOKEN (Code 190): Access token is invalid or expired. Refresh WHATSAPP_ACCESS_TOKEN with a permanent System User token.")
            elif err_code == 131030:
                logger.error("WhatsApp Cloud API TEST NUMBER RESTRICTION (Code 131030): Recipient %s is not registered in the Meta Developer Allowed Phone Numbers list.", to)
            elif err_code == 131047:
                logger.error("WhatsApp Cloud API WINDOW EXPIRED (Code 131047): More than 24 hours elapsed since customer's last message. Template message required.")
            else:
                logger.error(
                    "WhatsApp Cloud API send failed: status=%s code=%s message=%s",
                    response.status_code,
                    err_code,
                    err_msg
                )
            return False
        logger.info("WhatsApp text message successfully sent to %s", to)
        return True
    except Exception as e:
        logger.exception("Exception while sending WhatsApp message to %s: %s", to, e)
        return False


def download_whatsapp_media(media_id):
    """Fetch media metadata and binary bytes using Cloud API bearer token."""
    if not WHATSAPP_TOKEN:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN is not configured")

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "User-Agent": "DairyMateAI/1.0"
    }

    # 1. Fetch media URL
    meta_url = f"{GRAPH_BASE}/{media_id}"
    meta_resp = requests.get(meta_url, headers=headers, timeout=20)
    if meta_resp.status_code == 401:
        raise RuntimeError("WhatsApp Cloud API token unauthorized or expired (OAuthException 190).")
    meta_resp.raise_for_status()

    media_url = meta_resp.json().get("url")
    if not media_url:
        raise RuntimeError("WhatsApp media metadata did not contain a download URL")

    # 2. Download media binary bytes
    blob_resp = requests.get(media_url, headers=headers, timeout=30)
    blob_resp.raise_for_status()
    content = blob_resp.content

    if not content:
        raise RuntimeError("Downloaded WhatsApp media content was empty")
    if len(content) > 12 * 1024 * 1024:
        raise ValueError("Image exceeds the 12 MB maximum allowed size")

    return content


def format_result(kind, result):
    """Format AI screening prediction into clear, compassionate, actionable advice."""
    disclaimer = (
        "\n\n*Important:* Dairy Mate AI is a screening aid, not a veterinary "
        "diagnosis. Always consult a qualified veterinarian if you observe symptoms "
        "or have health concerns regarding your animal."
    )

    status = result.get("status")

    if status != "ok":
        reason = result.get("reason_key")
        if not reason and status == "use_udder_check":
            reason = "use_udder_check"
        elif not reason and status == "use_teat_check":
            reason = "use_teat_check"
        elif not reason:
            reason = "photo_not_suitable"

        friendly_reasons = {
            "photo_unreadable": (
                "⚠️ *Photo Unreadable*\n\n"
                "I couldn't process this image file. Please send a clear, uncorrupted photo."
            ),
            "too_dark": (
                "⚠️ *Photo Too Dark*\n\n"
                "This photo is too dark for reliable AI screening. Please take another photo in good daylight or lighting."
            ),
            "too_small": (
                "⚠️ *Photo Resolution Too Low*\n\n"
                "This image resolution is too low. Please send a higher-quality photo taken closer to the cow."
            ),
            "no_udder_found": (
                "⚠️ *Udder Not Detected*\n\n"
                "I couldn't identify an udder in this photo. Please send a clear photo showing the whole udder from a side or rear view."
            ),
            "no_teat_found": (
                "⚠️ *Teat Not Detected*\n\n"
                "I couldn't identify the teat clearly. Please send a clear close-up photo of the teat end."
            ),
            "use_teat_check": (
                "💡 *Suggestion: Teat Check*\n\n"
                "This photo looks like a close-up of a teat rather than a full udder view. "
                "Please reply with *teat* to run the Teat Check on this photo."
            ),
            "use_udder_check": (
                "💡 *Suggestion: Udder Check*\n\n"
                "This photo shows a full udder view rather than a teat close-up. "
                "Please reply with *udder* to run the Udder Check on this photo."
            ),
            "no_animal_found_scored_whole_photo": (
                "⚠️ *Animal Detection Uncertain*\n\n"
                "I couldn't clearly locate cattle in this photo. The result may be less reliable. "
                "For best results, photograph the cow in clear view."
            ),
            "udder_too_far": (
                "⚠️ *Udder Too Far Away*\n\n"
                "The cow is too far away in the frame. Please take a closer photo showing the udder clearly."
            ),
            "teat_too_far": (
                "⚠️ *Teat Too Far Away*\n\n"
                "The teat is too small or distant in the frame. Please take a closer photo of the teat end."
            ),
        }

        message = friendly_reasons.get(
            reason,
            "⚠️ I couldn't screen this photo reliably. Please try again with a clearer, well-lit photo."
        )
        return message + disclaimer

    label = result.get("label")
    confidence = result.get("confidence_level", "unknown")
    probability = result.get("probability")
    prob_str = f" ({probability * 100:.1f}% risk score)" if probability is not None else ""

    if confidence in ("low", "unknown", None):
        if kind == "udder":
            message = (
                "🔍 *Udder Check Result: Uncertain*\n\n"
                f"The AI screening result is inconclusive{prob_str}.\n\n"
                "Please inspect the animal physically for signs of mastitis:\n"
                "• Heat, swelling, redness, or hardness in the udder\n"
                "• Clots, flakes, watery appearance, or discoloration in milk\n"
                "• Tenderness or pain during milking\n\n"
                "If you observe any of these symptoms, contact a veterinarian immediately."
            )
        elif kind == "teat":
            message = (
                "🔍 *Teat Check Result: Uncertain*\n\n"
                f"The AI screening result is inconclusive{prob_str}.\n\n"
                "Please inspect the teat end for rough hyperkeratosis rings, lesions, or cracks. "
                "Consult a veterinarian if symptoms are present."
            )
        else:
            message = (
                "🔍 *Tick Check Result: Uncertain (Beta)*\n\n"
                "I couldn't determine tick presence with high confidence.\n\n"
                "Please manually inspect the animal's neck, dewlap, ears, groin, and tail base. "
                "Consult a veterinarian if ticks or irritation are found."
            )
        return message + disclaimer

    if kind == "udder":
        if label == "affected":
            message = (
                "⚠️ *Udder Check Result: Possible Abnormality Detected*\n\n"
                f"The AI screening flagged potential udder abnormalities (Confidence: {confidence.upper()}{prob_str}).\n\n"
                "Recommended Actions:\n"
                "1. Segregate the cow to prevent potential spread.\n"
                "2. Perform a California Mastitis Test (CMT) or strip cup check.\n"
                "3. Contact your local veterinarian for diagnosis and appropriate treatment."
            )
        else:
            message = (
                "✅ *Udder Check Result: No Clear Abnormality Detected*\n\n"
                f"The AI screening did not flag signs of mastitis in this photo (Confidence: {confidence.upper()}).\n\n"
                "Note: Early or subclinical conditions may not be visible in photos. Continue routine milking hygiene and health monitoring."
            )

    elif kind == "teat":
        if label == "affected":
            message = (
                "⚠️ *Teat Check Result: Possible Teat Condition Flagged*\n\n"
                f"The AI screening flagged potential teat abnormalities (Confidence: {confidence.upper()}{prob_str}).\n\n"
                "Recommended Actions:\n"
                "1. Check teat ends for hyperkeratosis rings, cracks, or milking machine damage.\n"
                "2. Check vacuum pressure and liner condition on milking equipment.\n"
                "3. Apply approved post-milking teat disinfectant.\n"
                "4. Consult a veterinarian if sores or inflammation persist."
            )
        else:
            message = (
                "✅ *Teat Check Result: No Clear Problem Detected*\n\n"
                f"The teat end appears healthy in this photo (Confidence: {confidence.upper()}).\n\n"
                "Continue using standard teat dipping and post-milking care."
            )

    else:  # tick
        if label == "affected":
            message = (
                "⚠️ *Tick Check Result: Ticks Detected (Beta)*\n\n"
                f"The AI screening detected probable tick clusters (Confidence: {confidence.upper()}{prob_str}).\n\n"
                "Recommended Actions:\n"
                "1. Physically examine the animal's coat, ears, brisket, and underbelly.\n"
                "2. Treat with approved acaricide / tick treatment recommended by your veterinarian.\n"
                "3. Check herd mates for possible infestation."
            )
        else:
            message = (
                "✅ *Tick Check Result: No Ticks Visible in Photo (Beta)*\n\n"
                f"The AI screening did not spot ticks in this photo area (Confidence: {confidence.upper()}).\n\n"
                "Please inspect the animal manually, as small or hidden ticks can be missed in photos."
            )

        if result.get("release_status") == "beta":
            message += (
                "\n\n_Note: Tick Check is in beta and has known limitations on close-up skin photos or field conditions._"
            )

    return message + disclaimer


def process_message(message):
    """Background processor for incoming WhatsApp messages."""
    sender = message.get("from")
    message_id = message.get("id")
    msg_type = message.get("type")

    if not sender:
        return

    logger.info("Processing message %s from %s (type=%s)", message_id, sender, msg_type)

    # 1. Handle TEXT messages
    if msg_type != "image":
        text_body = message.get("text", {}).get("body", "").strip()
        detected_kind = parse_model_kind(text_body)

        # Check if the user is replying with a model choice for a recently sent photo
        pending_media_id = None
        if detected_kind and sessions_collection is not None:
            session = sessions_collection.find_one({"_id": sender})
            if session:
                last_ts = session.get("updated_at")
                # Valid within 10 minutes
                if last_ts:
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - last_ts).total_seconds() < 600:
                        pending_media_id = session.get("pending_media_id")

        if pending_media_id and detected_kind:
            logger.info("User %s selected model '%s' for pending photo %s", sender, detected_kind, pending_media_id)
            execute_screening(sender, message_id, pending_media_id, detected_kind)
            # Clear pending media
            sessions_collection.update_one({"_id": sender}, {"$unset": {"pending_media_id": ""}})
            return

        # Otherwise send helpful usage guidance
        welcome_text = (
            "🐄 *Welcome to Dairy Mate AI* 🩺\n\n"
            "I provide AI-powered health screening aids for dairy cattle.\n\n"
            "*How to use:*\n"
            "1. Send a photo of the cow.\n"
            "2. Include a caption with the check you want:\n"
            "   • *udder* — Whole udder view (mastitis check)\n"
            "   • *teat* — Teat close-up view (teat condition check)\n"
            "   • *tick* — Animal coat view (tick check, beta)\n\n"
            "Or simply send a photo, and I will ask you which check to perform!\n\n"
            "_Note: Dairy Mate AI is a screening tool, not a veterinary diagnosis._"
        )
        send_text(sender, welcome_text)
        return

    # 2. Handle IMAGE messages
    image_info = message.get("image", {})
    media_id = image_info.get("id")
    caption = image_info.get("caption") or ""
    kind = parse_model_kind(caption)

    if not media_id:
        send_text(sender, "⚠️ I couldn't access the image attachment. Please try sending the photo again.")
        return

    # If no caption was provided, ask the user to select the model and save session state
    if not kind:
        if sessions_collection is not None:
            sessions_collection.update_one(
                {"_id": sender},
                {
                    "$set": {
                        "pending_media_id": media_id,
                        "updated_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )

        prompt_text = (
            "📸 *Photo received!*\n\n"
            "Which screening check would you like to run?\n\n"
            "Please reply with one of the following words:\n"
            "• *udder* — Whole udder view for mastitis screening\n"
            "• *teat* — Close-up view for teat end condition\n"
            "• *tick* — Animal body photo for tick detection (beta)\n\n"
            "_Dairy Mate AI is a screening aid, not a veterinary diagnosis._"
        )
        send_text(sender, prompt_text)

        if messages_collection is not None and message_id:
            messages_collection.update_one(
                {"_id": message_id},
                {"$set": {"status": "pending_user_selection", "media_id": media_id}}
            )
        return

    # Proceed with screening
    execute_screening(sender, message_id, media_id, kind)


def execute_screening(sender, message_id, media_id, kind):
    """Download photo, execute model inference, return formatted WhatsApp result, update MongoDB."""
    if messages_collection is not None and message_id:
        messages_collection.update_one(
            {"_id": message_id},
            {
                "$set": {
                    "status": "processing",
                    "model": kind,
                    "media_id": media_id,
                    "processing_started_at": datetime.now(timezone.utc)
                }
            },
            upsert=True
        )

    # Inform the user that processing is underway
    send_text(sender, f"🔄 Screening your photo with *{kind.capitalize()} Check*... Please wait a few seconds.")

    error_stage = None
    try:
        error_stage = "media_download"
        photo_bytes = download_whatsapp_media(media_id)

        error_stage = "model_inference"
        screener = get_screener(kind)
        result = screener.predict(photo_bytes)

        error_stage = "result_formatting"
        reply_message = format_result(kind, result)

        error_stage = "whatsapp_send"
        send_success = send_text(sender, reply_message)

        if messages_collection is not None and message_id:
            messages_collection.update_one(
                {"_id": message_id},
                {
                    "$set": {
                        "status": "completed" if send_success else "reply_failed",
                        "model": kind,
                        "prediction": result,
                        "reply_text": reply_message,
                        "completed_at": datetime.now(timezone.utc)
                    }
                }
            )

    except Exception as e:
        logger.exception("Inference workflow failed at stage '%s' for model '%s': %s", error_stage, kind, e)

        user_err_msg = (
            "⚠️ Sorry, I encountered an issue processing that photo.\n\n"
            "Please ensure the image is clear and well-lit, then try again. "
            "If the issue persists, our technical team is reviewing the issue."
        )
        send_text(sender, user_err_msg)

        if messages_collection is not None and message_id:
            messages_collection.update_one(
                {"_id": message_id},
                {
                    "$set": {
                        "status": "failed",
                        "error_stage": error_stage,
                        "error_message": str(e),
                        "failed_at": datetime.now(timezone.utc)
                    }
                }
            )


# ---------------------------------------------------------------------------
# HTTP ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    """Main health check endpoint for Render service."""
    return jsonify({
        "service": "Dairy Mate WhatsApp Backend",
        "status": "running"
    }), 200


@app.get("/models/health")
def models_health():
    """Health check for AI models."""
    return jsonify({
        "service": "Dairy Mate model inference",
        "available_models": ["udder", "teat", "tick"],
        "status": "ready"
    }), 200


@app.get("/system/status")
def system_status():
    """Diagnostic endpoint reporting backend, database, and credential status."""
    mongo_ok = False
    if db is not None:
        try:
            db.command("ping")
            mongo_ok = True
        except Exception:
            mongo_ok = False

    return jsonify({
        "service": "Dairy Mate WhatsApp Backend",
        "database": {
            "connected": mongo_ok,
            "database_name": MONGODB_DB_NAME
        },
        "meta_whatsapp_config": {
            "token_configured": bool(WHATSAPP_TOKEN),
            "phone_number_id_configured": bool(PHONE_NUMBER_ID),
            "verify_token_configured": bool(VERIFY_TOKEN),
            "app_secret_configured": bool(APP_SECRET),
            "graph_api_version": GRAPH_API_VERSION
        },
        "models": {
            "udder": "ready",
            "teat": "ready",
            "tick": "ready",
            "gate": "ready"
        }
    }), 200


@app.post("/models/predict")
def direct_predict():
    """Direct API testing endpoint for image inference (supports file upload or raw bytes)."""
    kind = request.form.get("model") or request.args.get("model")
    if kind not in ("udder", "teat", "tick"):
        return jsonify({"error": "model must be one of: udder, teat, tick"}), 400

    photo_bytes = None
    if "image" in request.files:
        photo_bytes = request.files["image"].read()
    elif request.data:
        photo_bytes = request.data

    if not photo_bytes:
        return jsonify({"error": "No image data provided. Send multipart form 'image' or raw body bytes."}), 400

    try:
        screener = get_screener(kind)
        result = screener.predict(photo_bytes)
        formatted_text = format_result(kind, result)
        return jsonify({
            "model": kind,
            "prediction": result,
            "formatted_message": formatted_text
        }), 200
    except Exception as e:
        logger.exception("Direct prediction error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.get("/webhook")
def verify_webhook():
    """Meta Webhook GET verification handshake."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and VERIFY_TOKEN and hmac.compare_digest(token or "", VERIFY_TOKEN) and challenge:
        logger.info("Meta webhook verification handshake successful.")
        return challenge, 200, {"Content-Type": "text/plain"}

    logger.warning("Meta webhook verification handshake failed: invalid token or parameters.")
    return "Verification failed", 403


@app.post("/webhook")
def receive_webhook():
    """Meta Webhook POST event handler."""
    raw_body = request.get_data()
    signature = request.headers.get("X-Hub-Signature-256", "")

    # Webhook signature validation if APP_SECRET is provided
    if APP_SECRET:
        if not signature or not signature.startswith("sha256="):
            logger.warning("Missing or malformed X-Hub-Signature-256 header.")
            return jsonify({"error": "Signature verification unavailable"}), 401

        expected = "sha256=" + hmac.new(
            APP_SECRET.encode("utf-8"),
            raw_body,
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            logger.warning("Invalid X-Hub-Signature-256 signature.")
            return jsonify({"error": "Invalid signature"}), 401

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON"}), 400

    # Parse entries and changes
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            field = change.get("field")
            value = change.get("value", {})
            messages = value.get("messages", [])

            logger.info(
                "WhatsApp webhook received: field=%s, messages=%d, statuses=%d",
                field,
                len(messages),
                len(value.get("statuses", []))
            )

            if field == "messages":
                for message in messages:
                    message_id = message.get("id")

                    # Deduplicate and store received message in MongoDB immediately
                    if message_id and messages_collection is not None:
                        try:
                            messages_collection.update_one(
                                {"_id": message_id},
                                {
                                    "$setOnInsert": {
                                        "message_id": message_id,
                                        "from": message.get("from"),
                                        "timestamp": message.get("timestamp"),
                                        "type": message.get("type"),
                                        "text": message.get("text", {}).get("body") if message.get("type") == "text" else None,
                                        "raw_message": message,
                                        "status": "received",
                                        "received_at": datetime.now(timezone.utc)
                                    }
                                },
                                upsert=True
                            )
                        except Exception as e:
                            logger.exception("Failed to store message %s in MongoDB: %s", message_id, e)

                    # Asynchronously process the message in background worker thread
                    executor.submit(process_message, message)

    # Acknowledge Meta immediately within <100ms to avoid webhook timeout
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
