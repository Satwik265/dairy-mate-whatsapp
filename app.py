import os
import json
import hmac
import hashlib
import logging

from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN", "")
APP_SECRET = os.environ.get("META_APP_SECRET", "")


@app.get("/")
def health():
    return jsonify({
        "service": "Dairy Mate WhatsApp Backend",
        "status": "running"
    }), 200


@app.get("/webhook")
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if (
        mode == "subscribe"
        and VERIFY_TOKEN
        and hmac.compare_digest(token or "", VERIFY_TOKEN)
        and challenge
    ):
        return challenge, 200, {"Content-Type": "text/plain"}

    return "Verification failed", 403


@app.post("/webhook")
def receive_webhook():
    raw_body = request.get_data()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not APP_SECRET or not signature.startswith("sha256="):
        return jsonify({
            "error": "Signature verification unavailable"
        }), 401

    expected = "sha256=" + hmac.new(
        APP_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256
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
            logging.info(
                "WhatsApp webhook received: field=%s, messages=%d, statuses=%d",
                change.get("field", "unknown"),
                len(value.get("messages", [])),
                len(value.get("statuses", []))
            )

    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
