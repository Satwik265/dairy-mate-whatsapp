"""
Local labelling tool for teat-end boxes. Python standard library only, nothing is uploaded anywhere.

    python scripts/label_teat_end.py          then open http://localhost:8765

Shows the photos from curation/teat_end_sample.csv one at a time. Draw a box around each teat end, press Enter.
Every save is written at once to curation/teat_end_labels.csv, so you can stop and come back any time;
it reopens at the first photo you haven't done. Guide: docs/teat_end_labelling.md.
"""

import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

TEAT = Path(__file__).resolve().parents[1]
SAMPLE = TEAT / "curation" / "teat_end_sample.csv"
LABELS = TEAT / "curation" / "teat_end_labels.csv"
DATA = TEAT / "data" / "processed_v2"
PAGE = Path(__file__).with_name("label_teat_end.html")
PORT = 8765
FIELDS = ["id", "status", "boxes", "labelled_at"]   # boxes: "x0 y0 x1 y1;..." as fractions of the photo
STATUSES = {"box", "no_teat_end", "unusable"}

lock = threading.Lock()


def load_labels():
    if not LABELS.exists():
        return {}
    return {r["id"]: r for r in csv.DictReader(open(LABELS, newline=""))}


def save_label(row):
    with lock:
        labels = load_labels()
        labels[row["id"]] = row
        tmp = LABELS.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(labels.values())
        tmp.replace(LABELS)


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/items":
            sample = list(csv.DictReader(open(SAMPLE, newline="")))
            labels = load_labels()
            items = [{"id": r["id"], "split": r["split"], "label": r["label"],
                      "label_row": labels.get(r["id"])} for r in sample]
            return self._send(json.dumps(items).encode())
        if self.path.startswith("/img/"):
            parts = unquote(self.path[5:]).split("/")          # split/label/id
            if len(parts) == 3 and all(p.replace("_", "").replace("-", "").isalnum() for p in parts):
                path = DATA / parts[0] / parts[1] / f"{parts[2]}.jpg"
                if path.exists():
                    return self._send(path.read_bytes(), "image/jpeg")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(b"not found", "text/plain", 404)
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if data.get("status") not in STATUSES:
            return self._send(b'{"error":"bad status"}', code=400)
        boxes = []
        for b in data.get("boxes", []):
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in b)
            boxes.append(f"{min(x0, x1):.4f} {min(y0, y1):.4f} {max(x0, x1):.4f} {max(y0, y1):.4f}")
        if data["status"] == "box" and not boxes:
            return self._send(b'{"error":"no box"}', code=400)
        save_label({"id": str(data["id"]), "status": data["status"], "boxes": ";".join(boxes),
                    "labelled_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._send(b'{"ok":true}')

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Teat-end labelling: open http://localhost:{PORT}   (Ctrl+C to stop; labels -> {LABELS})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
