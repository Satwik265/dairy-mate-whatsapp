import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as dm_app

TEST_IMAGES_DIR = Path(r"C:\Users\Admin\Downloads\DairyMate_WhatsApp_AI_Integration\DairyMate_models_code\DairyMate_Test_Images\DairyMate_Test_Images")


class TestImageFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dm_app.app.config["TESTING"] = True
        cls.client = dm_app.app.test_client()

    def test_01_image_screening_success_lifecycle(self):
        udder_path = TEST_IMAGES_DIR / "udder_01.png"
        with open(udder_path, "rb") as f:
            sample_bytes = f.read()

        msg_id = "wamid.test_img_001"
        sender = "919876543210"

        with patch("app.download_whatsapp_media", return_value=sample_bytes) as mock_dl, \
             patch("app.send_text", return_value=True) as mock_send:

            # Execute screening directly
            dm_app.execute_screening(sender, msg_id, "media_id_123", "udder")

            mock_dl.assert_called_once_with("media_id_123")
            # Should be called twice: 1 for "Screening your photo...", 1 for final result
            self.assertEqual(mock_send.call_count, 2)
            first_call_text = mock_send.call_args_list[0][0][1]
            second_call_text = mock_send.call_args_list[1][0][1]

            self.assertIn("Screening your photo with *Udder Check*", first_call_text)
            self.assertIn("Udder Check Result", second_call_text)

            # Check MongoDB record
            if dm_app.messages_collection is not None:
                doc = dm_app.messages_collection.find_one({"_id": msg_id})
                self.assertIsNotNone(doc)
                self.assertEqual(doc.get("status"), "completed")
                self.assertEqual(doc.get("model"), "udder")
                self.assertIn("prediction", doc)
                self.assertEqual(doc["prediction"].get("status"), "ok")
                # Clean up
                dm_app.messages_collection.delete_one({"_id": msg_id})

    def test_02_image_screening_failure_lifecycle(self):
        msg_id = "wamid.test_img_fail_001"
        sender = "919876543210"

        with patch("app.download_whatsapp_media", side_effect=RuntimeError("Token expired (code 190)")) as mock_dl, \
             patch("app.send_text", return_value=True) as mock_send:

            dm_app.execute_screening(sender, msg_id, "media_id_fail", "tick")

            # Check that error reply was sent
            mock_send.assert_called()
            last_text = mock_send.call_args[0][1]
            self.assertIn("Sorry, I encountered an issue", last_text)

            # Check MongoDB record
            if dm_app.messages_collection is not None:
                doc = dm_app.messages_collection.find_one({"_id": msg_id})
                self.assertIsNotNone(doc)
                self.assertEqual(doc.get("status"), "failed")
                self.assertEqual(doc.get("error_stage"), "media_download")
                self.assertIn("Token expired", doc.get("error_message"))
                # Clean up
                dm_app.messages_collection.delete_one({"_id": msg_id})

    def test_03_photo_without_caption_and_followup_text(self):
        udder_path = TEST_IMAGES_DIR / "udder_01.png"
        with open(udder_path, "rb") as f:
            sample_bytes = f.read()

        sender = "919811112222"
        img_msg = {
            "from": sender,
            "id": "wamid.test_no_caption_01",
            "type": "image",
            "image": {"id": "media_nocap_999", "caption": ""}
        }

        with patch("app.send_text", return_value=True) as mock_send:
            # Step 1: User sends photo without caption
            dm_app.process_message(img_msg)

            # Bot should prompt user to choose a check
            prompt_msg = mock_send.call_args[0][1]
            self.assertIn("Photo received", prompt_msg)
            self.assertIn("udder", prompt_msg)

            # Verify session was saved in MongoDB
            if dm_app.sessions_collection is not None:
                sess = dm_app.sessions_collection.find_one({"_id": sender})
                self.assertIsNotNone(sess)
                self.assertEqual(sess.get("pending_media_id"), "media_nocap_999")

        # Step 2: User replies with text "udder"
        text_msg = {
            "from": sender,
            "id": "wamid.test_text_choice_02",
            "type": "text",
            "text": {"body": "udder"}
        }

        with patch("app.download_whatsapp_media", return_value=sample_bytes) as mock_dl, \
             patch("app.send_text", return_value=True) as mock_send:

            dm_app.process_message(text_msg)

            # Bot should download media_nocap_999 and run Udder check
            mock_dl.assert_called_once_with("media_nocap_999")
            last_text = mock_send.call_args[0][1]
            self.assertIn("Udder Check Result", last_text)

            # Verify pending_media_id was cleared
            if dm_app.sessions_collection is not None:
                sess_after = dm_app.sessions_collection.find_one({"_id": sender})
                self.assertIsNone(sess_after.get("pending_media_id"))

        # Clean up test message records
        if dm_app.messages_collection is not None:
            dm_app.messages_collection.delete_one({"_id": "wamid.test_no_caption_01"})
            dm_app.messages_collection.delete_one({"_id": "wamid.test_text_choice_02"})
        if dm_app.sessions_collection is not None:
            dm_app.sessions_collection.delete_one({"_id": sender})


if __name__ == "__main__":
    unittest.main()

