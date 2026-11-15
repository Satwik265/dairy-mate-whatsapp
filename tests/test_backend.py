import os
import sys
import json
import hmac
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as dm_app
from model_api import get_screener

TEST_IMAGES_DIR = Path(r"C:\Users\Admin\Downloads\DairyMate_WhatsApp_AI_Integration\DairyMate_models_code\DairyMate_Test_Images\DairyMate_Test_Images")


class TestDairyMateBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dm_app.app.config["TESTING"] = True
        cls.client = dm_app.app.test_client()

    def test_01_health_endpoints(self):
        # GET /
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("service"), "Dairy Mate WhatsApp Backend")
        self.assertEqual(data.get("status"), "running")

        # GET /models/health
        res_m = self.client.get("/models/health")
        self.assertEqual(res_m.status_code, 200)
        data_m = res_m.get_json()
        self.assertIn("udder", data_m.get("available_models", []))
        self.assertIn("teat", data_m.get("available_models", []))
        self.assertIn("tick", data_m.get("available_models", []))

        # GET /system/status
        res_s = self.client.get("/system/status")
        self.assertEqual(res_s.status_code, 200)
        data_s = res_s.get_json()
        self.assertIn("database", data_s)
        self.assertIn("meta_whatsapp_config", data_s)
        self.assertIn("models", data_s)

    def test_02_webhook_verification(self):
        orig_token = dm_app.VERIFY_TOKEN
        dm_app.VERIFY_TOKEN = "test_verify_token_123"
        try:
            # Valid verify token
            res = self.client.get("/webhook?hub.mode=subscribe&hub.verify_token=test_verify_token_123&hub.challenge=test_challenge_abc")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_data(as_text=True), "test_challenge_abc")

            # Invalid verify token
            res_fail = self.client.get("/webhook?hub.mode=subscribe&hub.verify_token=wrong_token&hub.challenge=test_challenge_abc")
            self.assertEqual(res_fail.status_code, 403)
        finally:
            dm_app.VERIFY_TOKEN = orig_token

    def test_03_caption_parser(self):
        self.assertEqual(dm_app.parse_model_kind("udder"), "udder")
        self.assertEqual(dm_app.parse_model_kind("Udder check"), "udder")
        self.assertEqual(dm_app.parse_model_kind("teat inspection"), "teat")
        self.assertEqual(dm_app.parse_model_kind("Check for tick please"), "tick")
        self.assertIsNone(dm_app.parse_model_kind("hello cow"))
        self.assertIsNone(dm_app.parse_model_kind(""))
        self.assertIsNone(dm_app.parse_model_kind(None))

    def test_04_direct_predict_udder(self):
        udder_path = TEST_IMAGES_DIR / "udder_01.png"
        self.assertTrue(udder_path.exists(), f"Missing {udder_path}")
        with open(udder_path, "rb") as f:
            img_bytes = f.read()

        res = self.client.post(
            "/models/predict",
            data={"model": "udder", "image": (open(udder_path, "rb"), "udder_01.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("model"), "udder")
        self.assertIn("prediction", data)
        self.assertIn("formatted_message", data)
        pred = data["prediction"]
        self.assertEqual(pred.get("status"), "ok")
        self.assertIn(pred.get("label"), ("healthy", "affected"))

    def test_05_direct_predict_tick(self):
        tick_path = TEST_IMAGES_DIR / "tick_01.png"
        self.assertTrue(tick_path.exists(), f"Missing {tick_path}")
        res = self.client.post(
            "/models/predict",
            data={"model": "tick", "image": (open(tick_path, "rb"), "tick_01.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("model"), "tick")
        pred = data["prediction"]
        self.assertEqual(pred.get("status"), "ok")
        self.assertIn("release_status", pred)

    def test_06_direct_predict_teat_routing(self):
        # Sending whole udder image to teat check should route to use_udder_check
        udder_path = TEST_IMAGES_DIR / "udder_01.png"
        res = self.client.post(
            "/models/predict",
            data={"model": "teat", "image": (open(udder_path, "rb"), "udder_01.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("model"), "teat")
        pred = data["prediction"]
        self.assertEqual(pred.get("status"), "use_udder_check")
        self.assertIn("Udder Check", data["formatted_message"])

    def test_06b_direct_predict_teat_classification(self):
        # Sending teat close-up image to teat check should produce real ok status and label
        teat_path = TEST_IMAGES_DIR / "udder_05.png"
        self.assertTrue(teat_path.exists(), f"Missing {teat_path}")
        res = self.client.post(
            "/models/predict",
            data={"model": "teat", "image": (open(teat_path, "rb"), "udder_05.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("model"), "teat")
        pred = data["prediction"]
        self.assertEqual(pred.get("status"), "ok")
        self.assertIn(pred.get("label"), ("healthy", "affected"))
        self.assertIsNotNone(pred.get("probability"))
        self.assertIn("Teat Check Result", data["formatted_message"])

    def test_07_post_webhook_with_signature(self):
        orig_secret = dm_app.APP_SECRET
        dm_app.APP_SECRET = "unit_test_secret"
        try:
            payload = json.dumps({
                "object": "whatsapp_business_account",
                "entry": [{
                    "id": "12345",
                    "changes": [{
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "123456", "phone_number_id": "987654"},
                            "messages": [{
                                "from": "15551234567",
                                "id": "wamid.unit_test_001",
                                "timestamp": "1700000000",
                                "type": "text",
                                "text": {"body": "hello"}
                            }]
                        }
                    }]
                }]
            }).encode("utf-8")

            # Missing signature should fail with 401
            res_no_sig = self.client.post("/webhook", data=payload, content_type="application/json")
            self.assertEqual(res_no_sig.status_code, 401)

            # Valid signature should succeed with 200
            valid_sig = "sha256=" + hmac.new(dm_app.APP_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
            res_valid = self.client.post(
                "/webhook",
                data=payload,
                headers={"X-Hub-Signature-256": valid_sig},
                content_type="application/json"
            )
            self.assertEqual(res_valid.status_code, 200)
            self.assertEqual(res_valid.get_json().get("status"), "received")
        finally:
            dm_app.APP_SECRET = orig_secret


if __name__ == "__main__":
    unittest.main()

