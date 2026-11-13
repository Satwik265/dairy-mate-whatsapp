import os
import json
import hmac
import hashlib
import logging
from functools import lru_cache

import requests
from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dairy_mate")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing from .env")

mongo_client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000,
)

db = mongo_client[MONGODB_DB_NAME]
messages_collection = db["whatsapp_messages"]
app = Flask(__name__)
@app.get("/models/health")
def models_health():
    return jsonify({
        "service": "Dairy Mate model inference",
        "available_models": ["udder", "teat", "tick"],
        "status": "ready_to_test"
    }), 200
logging.basicConfig(level=logging.INFO)

VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN", "")
APP_SECRET = os.environ.get("META_APP_SECRET", "")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
GRAPH_API_VERSION = os.environ.get("WHATSAPP_GRAPH_API_VERSION", "v26.0")
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

from model_api import get_screener  # noqa: E402


def send_text(to, body):
    """Send a WhatsApp text reply. Requires a valid Cloud API token and phone number ID."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        logging.error("Cannot send WhatsApp reply: Cloud API credentials are not configured")
        return False
    response = requests.post(
        f"{GRAPH_BASE}/{PHONE_NUMBER_ID}/messages",
        headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"},
        json={
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": body[:4000]},
        },
        timeout=20,
    )
    if not response.ok:
        logging.error("WhatsApp send failed: status=%s body=%s", response.status_code, response.text[:500])
        return False
    return True


def download_whatsapp_media(media_id):
    """Fetch media metadata and bytes using the Cloud API bearer token."""
    if not WHATSAPP_TOKEN:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN is not configured")
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    meta = requests.get(f"{GRAPH_BASE}/{media_id}", headers=headers, timeout=20)
    meta.raise_for_status()
    media_url = meta.json().get("url")
    if not media_url:
        raise RuntimeError("WhatsApp media URL missing")
    blob = requests.get(media_url, headers=headers, timeout=30)
    blob.raise_for_status()
    content = blob.content
    if not content:
        raise RuntimeError("WhatsApp media was empty")
    if len(content) > 12 * 1024 * 1024:
        raise ValueError("Image exceeds 12 MB limit")
    return content




def format_result(kind, result):
    disclaimer = (
        "\n\nImportant: This is an AI screening aid, not a veterinary "
        "diagnosis. Please consult a veterinarian if you notice symptoms "
        "or remain concerned."
    )

    if result.get("status") != "ok":
        reason = result.get("reason_key")

        if not reason and result.get("status") == "use_udder_check":
            reason = "use_udder_check"
        elif not reason:
            reason = "photo_not_suitable"

        friendly = {
            "photo_unreadable": (
                "I couldn't read this photo. Please send a clearer image."
            ),
            "too_dark": (
                "This photo is too dark. Please take another photo in good light."
            ),
            "too_small": (
                "This photo is too small. Please send a higher-quality image."
            ),
            "no_udder_found": (
                "I couldn't identify the udder. Please send a clear photo "
                "showing the whole udder."
            ),
            "no_teat_found": (
                "I couldn't identify the teat clearly. Please send a close-up "
                "photo of the teat."
            ),
            "use_teat_check": (
                "This photo appears more suitable for a teat check. "
                "Please resend it with the caption: teat"
            ),
            "use_udder_check": (
                "This photo appears more suitable for an udder check. "
                "Please resend it with the caption: udder"
            ),
            "no_animal_found_scored_whole_photo": (
                "I couldn't clearly identify an animal in this photo. "
                "The result may be unreliable. Please send a clearer photo."
            ),
        }

        message = friendly.get(
            reason,
            "I couldn't screen this photo reliably. Please try a clearer photo.",
        )
        return message + disclaimer

    label = result.get("label")
    confidence = result.get("confidence_level", "unknown")

    if confidence in ("low", "unknown", None):
        if kind == "udder":
            message = (
                "Udder Check\n\n"
                "The result is uncertain. I couldn't reliably determine "
                "whether this photo shows an udder problem.\n\n"
                "Please check for swelling, unusual warmth, pain, or changes "
                "in the milk. If you notice these signs, contact a veterinarian."
            )
        elif kind == "teat":
            message = (
                "Teat Check\n\n"
                "The result is uncertain. Please send a clear close-up photo "
                "of the teat, or consult a veterinarian if you notice a problem."
            )
        else:
            message = (
                "Tick Check\n\n"
                "I couldn't reliably determine whether ticks are present. "
                "Please inspect the animal carefully and seek veterinary "
                "advice if you suspect an infestation."
            )
        return message + disclaimer

    if kind == "udder":
        if label == "affected":
            message = (
                "Udder Check\n\n"
                "The AI screening detected possible udder abnormalities. "
                "Please check the animal and contact a veterinarian for advice."
            )
        else:
            message = (
                "Udder Check\n\n"
                "The AI screening did not flag a clear abnormality in this photo. "
                "This does not rule out disease. Monitor the animal for symptoms."
            )

    elif kind == "teat":
        if label == "affected":
            message = (
                "Teat Check\n\n"
                "The AI screening flagged a possible teat problem. "
                "Please have the animal checked by a veterinarian."
            )
        else:
            message = (
                "Teat Check\n\n"
                "The AI screening did not flag a clear teat problem in this photo. "
                "If symptoms are present, seek veterinary advice."
            )

    else:
        if label == "affected":
            message = (
                "Tick Check\n\n"
                "The AI screening detected possible ticks. Please inspect "
                "the animal carefully and ask a veterinarian about appropriate "
                "treatment."
            )
        else:
            message = (
                "Tick Check\n\n"
                "The AI screening did not flag ticks in this photo. "
                "Please still inspect the animal, as ticks may be missed."
            )

        if result.get("release_status") == "beta":
            message += (
                "\n\nTick screening is in beta and may miss ticks, "
                "especially in close-up photos or unfamiliar farm conditions."
            )

    return message + disclaimer




def process_message(message):
    sender = message.get("from")
    if not sender:
        return
    if message.get("type") != "image":
        send_text(sender, "Welcome to Dairy Mate AI. To screen a photo, send an image with a caption naming the check: udder, teat, or tick. These are screening aids, not veterinary diagnoses.")
        return

    image_info = message.get("image", {})
    media_id = image_info.get("id")
    caption = (image_info.get("caption") or "").strip().lower()
    # Require explicit selection; never silently guess which classifier a farmer intended.
    kind = next((k for k in ("udder", "teat", "tick") if caption == k or caption.startswith(k + " ")), None)
    if not kind:
        send_text(sender, "Please resend the photo with one caption: udder, teat, or tick. Tick Check is beta and can miss ticks, especially in close-up photos.")
        return
    if not media_id:
        send_text(sender, "I couldn't access the image attachment. Please send the photo again.")
        return

    try:
        photo_bytes = download_whatsapp_media(media_id)
        result = get_screener(kind).predict(photo_bytes)
        send_text(sender, format_result(kind, result))
    except Exception:
        logging.exception("Failed to process incoming WhatsApp image (model=%s)", kind)
        send_text(sender, "Sorry, I couldn't process that photo right now. Please try again later or contact the Dairy Mate team.")


@app.get("/")
def health():
    return jsonify({"service": "Dairy Mate WhatsApp Backend", "status": "running"}), 200


@app.get("/webhook")
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and VERIFY_TOKEN and hmac.compare_digest(token or "", VERIFY_TOKEN) and challenge:
        return challenge, 200, {"Content-Type": "text/plain"}
    return "Verification failed", 403


@app.post("/webhook")
def receive_webhook():
    raw_body = request.get_data()
    signature = request.headers.get("X-Hub-Signature-256", "")

    # Keep Meta signature verification.
    if not APP_SECRET or not signature.startswith("sha256="):
        return jsonify({"error": "Signature verification unavailable"}), 401

    expected = "sha256=" + hmac.new(
        APP_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        return jsonify({"error": "Invalid signature"}), 401

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON"}), 400

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])

            logging.info(
                "WhatsApp webhook received: field=%s, messages=%d, statuses=%d",
                change.get("field", "unknown"),
                len(messages),
                len(value.get("statuses", [])),
            )

            if change.get("field") == "messages":
                for message in messages:
                    message_id = message.get("id")

                    # Store only identifiable WhatsApp messages.
                    if message_id:
                        try:
                            messages_collection.update_one(
                                {"_id": message_id},
                                {
                                    "$setOnInsert": {
                                        "message_id": message_id,
                                        "from": message.get("from"),
                                        "timestamp": message.get("timestamp"),
                                        "type": message.get("type"),
                                        "text": message.get("text", {}).get("body")
                                        if message.get("type") == "text"
                                        else None,
                                        "raw_message": message,
                                        "status": "received",
                                    }
                                },
                                upsert=True,
                            )
                        except Exception:
                            logging.exception(
                                "Failed to store WhatsApp message %s",
                                message_id,
                            )
                            # Return a failure so Meta can retry delivery.
                            return jsonify({"error": "Message storage failed"}), 500

                    process_message(message)

    return jsonify({"status": "received"}), 200



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
