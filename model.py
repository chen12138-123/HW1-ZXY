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


def _spatial_color_hist(rgb: np.ndarray, bins: int = 16, grid: int = 2) -> np.ndarray:
    h, w, _ = rgb.shape
    feats = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = int(round(gy * h / grid))
            y1 = int(round((gy + 1) * h / grid))
            x0 = int(round(gx * w / grid))
            x1 = int(round((gx + 1) * w / grid))
            patch = rgb[y0:y1, x0:x1, :]
            feats.append(_color_hist(patch, bins=bins))
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


def _hog(gray: np.ndarray, cell_size: int = 8, block_size: int = 2, num_bins: int = 9) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0

    h, w = g.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    if n_cells_y < 1 or n_cells_x < 1:
        return np.zeros((0,), dtype=np.float32)

    mag = mag[: n_cells_y * cell_size, : n_cells_x * cell_size]
    ang = ang[: n_cells_y * cell_size, : n_cells_x * cell_size]

    bin_width = 180.0 / num_bins
    cell_hist = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            y1 = y0 + cell_size
            x0 = cx * cell_size
            x1 = x0 + cell_size
            a = ang[y0:y1, x0:x1]
            m = mag[y0:y1, x0:x1]
            bins = np.floor(a / bin_width).astype(np.int32)
            bins = np.clip(bins, 0, num_bins - 1)
            for b in range(num_bins):
                cell_hist[cy, cx, b] = m[bins == b].sum()

    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    if n_blocks_y < 1 or n_blocks_x < 1:
        return cell_hist.reshape(-1)

    blocks = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_hist[by : by + block_size, bx : bx + block_size, :].reshape(-1)
            norm = np.sqrt((block * block).sum() + 1e-6)
            blocks.append(block / norm)
    return np.concatenate(blocks, axis=0)


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


def extract_features_from_rgb(
    rgb: np.ndarray,
    color_bins: int = 16,
    edge_bins: int = 16,
    feature_set: str = "basic",
    hog_cell: int = 8,
    hog_block: int = 2,
    hog_bins: int = 9,
) -> np.ndarray:
    gray = (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).astype(np.float32)
    fs = (feature_set or "basic").lower()
    feats = []

    if fs in {"basic", "basic_hog", "hog"}:
        feats.append(_color_hist(rgb, bins=color_bins))
        feats.append(_lbp_hist(gray))
        feats.append(_edge_hist(gray, bins=edge_bins))

    if fs in {"spatial", "spatial_hog"}:
        feats.append(_color_hist(rgb, bins=color_bins))
        feats.append(_spatial_color_hist(rgb, bins=color_bins, grid=2))
        feats.append(_lbp_hist(gray))
        feats.append(_edge_hist(gray, bins=edge_bins))

    if fs in {"hog", "basic_hog", "spatial_hog"}:
        feats.append(_hog(gray, cell_size=hog_cell, block_size=hog_block, num_bins=hog_bins))

    if not feats:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    return np.concatenate(feats, axis=0)


def extract_features(
    path: str,
    image_size: int = 64,
    color_bins: int = 16,
    edge_bins: int = 16,
    feature_set: str = "basic",
    hog_cell: int = 8,
    hog_block: int = 2,
    hog_bins: int = 9,
) -> np.ndarray:
    rgb = load_rgb_image(path, image_size=image_size)
    return extract_features_from_rgb(
        rgb,
        color_bins=color_bins,
        edge_bins=edge_bins,
        feature_set=feature_set,
        hog_cell=hog_cell,
        hog_block=hog_block,
        hog_bins=hog_bins,
    )


def create_classifier(model_name: str = "linear_svm", seed: int = 42):
    name = (model_name or "").lower()
    if name in {"linear_svm", "svm", "linearsvc"}:
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LinearSVC(C=4.0, class_weight="balanced", random_state=seed, max_iter=8000, dual=False)),
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
