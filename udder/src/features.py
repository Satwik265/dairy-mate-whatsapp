"""
Feature extraction: CLAHE contrast normalization, HSV/BGR color stats,
LBP + GLCM texture, Canny edge density, HOG shape. 223-dim vector.
Proven approach reused from the earlier mastitis (udder) project.
"""
import csv
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
IMG_SIZE = 128


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_features(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = apply_clahe(gray)

    feats = []
    for ch in range(3):
        h_ = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
        h_ = cv2.normalize(h_, h_).flatten()
        feats.extend(h_)

    for img_space in (img, hsv):
        for ch in range(3):
            channel = img_space[:, :, ch].astype(np.float32)
            feats.append(channel.mean())
            feats.append(channel.std())

    lbp = local_binary_pattern(gray_eq, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    feats.extend(lbp_hist)

    gray_q = (gray_eq / 8).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop)
        feats.append(vals.mean())
        feats.append(vals.std())

    edges = cv2.Canny(gray_eq, 100, 200)
    feats.append(edges.mean() / 255.0)

    hog_feats = hog(gray_eq, orientations=9, pixels_per_cell=(32, 32),
                     cells_per_block=(1, 1), feature_vector=True)
    feats.extend(hog_feats)

    return np.array(feats, dtype=np.float32)


def load_manifest(csv_path):
    return list(csv.DictReader(open(csv_path)))


if __name__ == "__main__":
    rows = load_manifest(SPLITS_DIR / "annotations_full.csv")
    X, y = [], []
    for row in rows:
        img = cv2.imread(row["filepath"])
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(int(row["label"]))
    X = np.vstack(X)
    print(f"Feature vector length: {X.shape[1]}, shape: {X.shape}")
