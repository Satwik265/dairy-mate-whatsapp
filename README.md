# Dairy Mate AI — WhatsApp Backend & Health Screening System

AI-powered mobile screening aids for dairy cattle health (Udder mastitis screening, Teat condition analysis, and Tick detection) deployed on WhatsApp Cloud API, Flask, and Render.

> [!IMPORTANT]
> **Veterinary Disclaimer**: Dairy Mate AI models are computer-vision screening aids designed for early detection and herd monitoring. They do **not** constitute a veterinary diagnosis. Always consult a qualified veterinary professional for diagnosis and clinical treatment plans.

---

## Capabilities

| Screener | Pipeline Architecture | Input Specs | Output & Confidence |
|---|---|---|---|
| **Udder Check** | Shared YOLO11n Photo Gate + MobileNetV2 feature classifier + Logistic Regression | Whole udder photo | Healthy vs. Affected (Probabilistic risk score, High/Medium/Low confidence) |
| **Teat Check** | Shared YOLO11n Photo Gate + Fine-tuned MobileNetV2 + Platt Calibration | Teat close-up photo | Healthy vs. Teat condition detected (Hyperkeratosis / damage screening) |
| **Tick Check (Beta)** | YOLO11n Animal Detector + MobileNetV2 96x96 patch classifier + Top-K aggregation | Animal coat / body photo | Ticks detected vs. No ticks visible in photo area (Beta release) |

---

## Production Architecture

- **Backend**: Flask 3.1 with Gunicorn on Render (`https://dairy-mate-whatsapp-11.onrender.com`)
- **Messaging**: WhatsApp Cloud API (Graph API `v21.0`) with HMAC SHA-256 signature verification (`X-Hub-Signature-256`)
- **Asynchronous Execution**: Immediate HTTP 200 webhook acknowledgment with background thread pool execution to eliminate timeout drops
- **Session Memory**: Two-step photo screening support (if photo sent without caption, bot prompts user and processes follow-up keyword)
- **Database**: MongoDB Atlas (`whatsapp_messages` and `user_sessions`) for audit trails, deduplication, and failure tracking
- **Inference Runtime**: ONNX Runtime CPU (`onnxruntime`) with background model pre-warming on container startup

---

## WhatsApp Interaction Flow

1. **Text Message**: Sending any greeting or text receives a menu explaining how to screen cattle photos.
2. **Photo with Caption**: Send photo with caption `udder`, `teat`, or `tick` for instant automated screening.
3. **Photo without Caption**: Send any photo; Dairy Mate AI will acknowledge the photo and prompt you to choose `udder`, `teat`, or `tick`. Replying with the keyword triggers immediate screening.

---

## Quick Start (Local Development)

```bash
# 1. Clone repository
git clone https://github.com/Satwik265/dairy-mate-whatsapp.git
cd dairy-mate-whatsapp

# 2. Virtual environment setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run test suite
python -m unittest discover -s tests

# 5. Run local development server
python app.py
```

---

## Detailed Operations Guide

For complete configuration instructions, Meta Developer portal settings, permanent System User tokens, and troubleshooting, see [INTEGRATION_README.md](INTEGRATION_README.md).
