import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_rgb_image(path: str, image_size: int = 64) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _color_hist(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    feats = []
    for c in range(3):
        h, _ = np.histogram(rgb[..., c], bins=bins, range=(0.0, 1.0), density=False)
        h = h.astype(np.float32)
        h /= max(h.sum(), 1.0)
        feats.append(h)
    return np.concatenate(feats, axis=0)


def _lbp_hist(gray: np.ndarray) -> np.ndarray:
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    c = gray_u8[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    code |= (gray_u8[0:-2, 0:-2] >= c).astype(np.uint8) << 7
    code |= (gray_u8[0:-2, 1:-1] >= c).astype(np.uint8) << 6
    code |= (gray_u8[0:-2, 2:  ] >= c).astype(np.uint8) << 5
    code |= (gray_u8[1:-1, 2:  ] >= c).astype(np.uint8) << 4
    code |= (gray_u8[2:  , 2:  ] >= c).astype(np.uint8) << 3
    code |= (gray_u8[2:  , 1:-1] >= c).astype(np.uint8) << 2
    code |= (gray_u8[2:  , 0:-2] >= c).astype(np.uint8) << 1
    code |= (gray_u8[1:-1, 0:-2] >= c).astype(np.uint8) << 0
    hist = np.bincount(code.reshape(-1), minlength=256).astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    return hist


def _edge_hist(gray: np.ndarray, bins: int = 16) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.abs(g[:, 2:] - g[:, :-2])
    gy = np.abs(g[2:, :] - g[:-2, :])
    gx = np.pad(gx, ((0, 0), (1, 1)), mode="edge")
    gy = np.pad(gy, ((1, 1), (0, 0)), mode="edge")
    mag = np.sqrt(gx * gx + gy * gy)
    h, _ = np.histogram(mag, bins=bins, range=(0.0, float(mag.max() if mag.size else 1.0)), density=False)
    h = h.astype(np.float32)
    h /= max(h.sum(), 1.0)
    return h


def extract_features_from_rgb(rgb: np.ndarray, color_bins: int = 16, edge_bins: int = 16) -> np.ndarray:
    gray = (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).astype(np.float32)
    f1 = _color_hist(rgb, bins=color_bins)
    f2 = _lbp_hist(gray)
    f3 = _edge_hist(gray, bins=edge_bins)
    return np.concatenate([f1, f2, f3], axis=0)


def extract_features(path: str, image_size: int = 64, color_bins: int = 16, edge_bins: int = 16) -> np.ndarray:
    rgb = load_rgb_image(path, image_size=image_size)
    return extract_features_from_rgb(rgb, color_bins=color_bins, edge_bins=edge_bins)


def create_classifier(model_name: str = "linear_svm", seed: int = 42):
    name = (model_name or "").lower()
    if name in {"linear_svm", "svm", "linearsvc"}:
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LinearSVC(C=2.0, class_weight="balanced", random_state=seed)),
            ]
        )
    if name in {"logreg", "logistic"}:
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=2000, n_jobs=None, multi_class="auto", random_state=seed)),
            ]
        )
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
