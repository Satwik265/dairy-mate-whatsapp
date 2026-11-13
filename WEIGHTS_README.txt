DairyMate model weights (2026-10-05, matching code commit 4b228d6)
=====================================================================

Unzip this into the same folder as DairyMate_models_code: the paths below match, so every predict.py finds
its weights without any changes.

Shared
  common/models/photo_gate.onnx            YOLO11n photo gate: finds a whole udder or a teat close-up and routes
                                           the photo (used by Udder Check and Teat Check)

Udder Check (udder/src/predict.py)
  udder/models/udder_classifier.keras      MobileNetV2 features + logistic regression + calibration (Keras 3.15 / TF 2.21)
  udder/models/udder_classifier.onnx       same model converted to ONNX (used by the website, no TensorFlow needed)

Teat Check (teat/scripts/predict.py)
  teat/models/teat_classifier_v2.keras     fine-tuned MobileNetV2, v2 (Keras 3.15 / TF 2.21)
  teat/models/teat_classifier_v2.onnx      same model converted to ONNX (used by the website)

Tick Check, Beta (tick/scripts/predict.py)
  tick/models/tick_patch_B.onnx            MobileNetV2 96x96 patch classifier, variant B (what predict.py runs)
  tick/models/tick_patch_B.pt              same model, PyTorch checkpoint (for further training)
  tick/models/animal_detector.onnx         stock COCO YOLO11n, used only to crop to the animal (not trained on ticks)

The ONNX copies give exactly the same status, verdict and probability as the Keras files on all held-out test
photos (353 udder, 312 teat): web/reports/onnx_parity.json in the repository.

Calibration, confidence bands and limitations for each model are in its model card (models/*_model_card.json)
and README in the code zip. Every model has been validated only on web/research images, not on field photos.
They are screening aids, not a veterinary diagnosis.

Quick test (TensorFlow venv, Python 3.12, from the folder holding both unzipped parts):
  python udder/src/predict.py photo.jpg
  python teat/scripts/predict.py photo.jpg
  python tick/scripts/predict.py photo.jpg
