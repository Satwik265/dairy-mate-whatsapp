# Dairy Mate WhatsApp + AI model integration

This project combines the supplied model code and matching weights with the existing Flask Meta webhook.
It preserves GET webhook verification and validates `X-Hub-Signature-256` on POST requests.

## What the WhatsApp flow does

- A WhatsApp text message receives instructions.
- An incoming photo must have the caption `udder`, `teat`, or `tick` so the bot does not guess the intended check.
- The server downloads the WhatsApp media using the Cloud API token, runs the supplied predictor, and sends a screening summary.
- Tick Check is labelled beta and its limitations are mentioned in the reply.
- All results carry a screening-aid / not-a-diagnosis disclaimer.

## Required environment variables

Set these in your local shell or Render Environment settings (never commit them):

- `META_VERIFY_TOKEN` — must match the token in the Meta webhook configuration.
- `META_APP_SECRET` — app secret from Meta App Settings; used to verify webhook signatures.
- `WHATSAPP_ACCESS_TOKEN` — a valid WhatsApp Cloud API access token with permission to read media and send messages.
- `WHATSAPP_PHONE_NUMBER_ID` — Phone Number ID for the WhatsApp sender registered with Cloud API.
- `WHATSAPP_GRAPH_API_VERSION` — optional; defaults to `v26.0` to match the currently selected dashboard version.
- `DM_ORT_THREADS=1` and `DM_ORT_ARENA=0` — recommended for a low-memory Render instance; benchmark before production.

## Local test

1. Use Python 3.12 if possible.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Set the four required environment variables above.
5. Start the server: `gunicorn --bind 127.0.0.1:5000 app:app`.
6. `GET /` should return `{"service":"Dairy Mate WhatsApp Backend","status":"running"}`.
7. Send representative images directly to the predictor scripts first:
   - `python udder/src/predict.py sample_udder.jpg`
   - `python teat/scripts/predict.py sample_teat.jpg`
   - `python tick/scripts/predict.py sample_tick.jpg`
8. After deployment, use a phone allowed by Meta's test-number setup and send an image with the caption `udder`, `teat`, or `tick`.

## Important deployment caveats

- This has not been tested against live Meta credentials from this workspace; use Meta's test number first.
- The webhook processes image inference synchronously. If Render/Meta times out, move processing to a queue and acknowledge the webhook quickly.
- First inference loads a model and may be slow; each worker loads its own copy. Start with one worker and check memory before increasing workers.
- The supplied model documentation says the models have only been validated on web/research images, not farm field photos. Tick Check is particularly weak on close-up / other-farm photos and must not be presented as reliable tick exclusion.
- Udder/teat results are screening aids only, not a veterinary diagnosis. Teat output must not be called a mastitis diagnosis.
- Never put tokens, app secrets, verify tokens, OTPs, or customer photos in GitHub commits or logs.
