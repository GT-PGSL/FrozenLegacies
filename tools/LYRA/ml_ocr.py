"""
ml_ocr.py -- ML-based digit classifier for LYRA CBD recognition
================================================================
Replaces (or augments) structural 7-segment OCR with a RandomForest
trained on labeled digit images from processed flights.

Training:
    python tools/LYRA/ml_ocr.py --train --flights 125,126,127,128

Evaluation (stratified hold-out):
    python tools/LYRA/ml_ocr.py --evaluate --flights 125,126,127,128

Single-TIFF prediction (standalone test):
    python tools/LYRA/ml_ocr.py --predict Data/ascope/raw/128/46_0000000_0000024.tiff

Production API (imported by detect_frames.py):
    from ml_ocr import recognize_frame_ml, load_model
"""

from pathlib import Path
import sys
import csv
import argparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/LYRA"))
from lyra import detect_frames
from build_digit_templates import (
    get_binary_text, find_digit_col_ranges, adjust_blobs_to_7,
    TMPL_W, TMPL_H,
)
from segment_ocr import (
    crop_and_resize_digit, measure_segment_densities,
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "digit_rf.pkl"


# =========================================================================
# Feature extraction
# =========================================================================

def extract_features(digit_img, orig_aspect):
    """
    Extract a 24-element feature vector from a digit image.

    Features (24 total):
        [0:7]   - 7 segment densities (a-g) from measure_segment_densities()
        [7:15]  - 8 horizontal projection bins (row sums split into 8 bands)
        [15:20] - 5 vertical projection bins (col sums split into 5 bands)
        [20]    - aspect ratio (orig blob width / median blob width)
        [21]    - total pixel density (mean of all pixels)
        [22:24] - ink centroid (x, y) normalized to [0, 1]

    Parameters
    ----------
    digit_img : np.ndarray, float32
        (TMPL_H, TMPL_W) binary image (1=ink, 0=bg).
    orig_aspect : float
        Original blob width / median blob width.

    Returns
    -------
    features : np.ndarray, float64, shape (24,)
    """
    H, W = digit_img.shape

    # -- Segment densities (7) --
    densities, _, _ = measure_segment_densities(digit_img, orig_aspect)
    seg_feats = [densities[s] for s in ["a", "b", "c", "d", "e", "f", "g"]]

    # -- Horizontal projection bins (8) --
    row_proj = digit_img.mean(axis=1)  # shape (H,)
    n_hbins = 8
    hbin_size = H / n_hbins
    h_feats = []
    for i in range(n_hbins):
        r0 = int(i * hbin_size)
        r1 = int((i + 1) * hbin_size)
        h_feats.append(float(row_proj[r0:r1].mean()))

    # -- Vertical projection bins (5) --
    col_proj = digit_img.mean(axis=0)  # shape (W,)
    n_vbins = 5
    vbin_size = W / n_vbins
    v_feats = []
    for i in range(n_vbins):
        c0 = int(i * vbin_size)
        c1 = int((i + 1) * vbin_size)
        v_feats.append(float(col_proj[c0:c1].mean()))

    # -- Scalar features --
    aspect_feat = float(orig_aspect) if orig_aspect is not None else 1.0
    density_feat = float(digit_img.mean())

    # -- Ink centroid (x, y) normalized --
    total_ink = digit_img.sum()
    if total_ink > 0:
        rows, cols = np.where(digit_img > 0.1)
        cy = float(rows.mean()) / H
        cx = float(cols.mean()) / W
    else:
        cx, cy = 0.5, 0.5

    features = np.array(
        seg_feats + h_feats + v_feats + [aspect_feat, density_feat, cx, cy],
        dtype=np.float64,
    )
    return features


# =========================================================================
# Training data loading
# =========================================================================

def _find_tiff(raw_dir, tiff_id):
    """Find TIFF file matching a tiff_id in a raw directory."""
    # tiff_id could be "7700", "0", "8525", etc.
    tid_padded = f"{int(tiff_id):07d}"
    matches = list(raw_dir.glob(f"*_{tid_padded}_*-reel_begin_end.tiff"))
    if not matches:
        matches = list(raw_dir.glob(f"*_{tid_padded}_*"))
    if not matches:
        # Try unpadded
        matches = list(raw_dir.glob(f"*_{tiff_id}_*"))
    return matches[0] if matches else None


def load_training_data(flights):
    """
    Load labeled digit images from frame_index.csv files.

    For each flight, reads the corrected frame_index.csv, loads each TIFF,
    extracts digit blobs, pairs with ground truth CBD labels.

    Parameters
    ----------
    flights : list of int
        Flight numbers to load (e.g. [125, 126, 127, 128]).

    Returns
    -------
    X : np.ndarray, shape (n_samples, 24)
    y : np.ndarray, shape (n_samples,) of int (0-9)
    metadata : list of dict with tiff_id, frame_idx, pos, flight
    """
    X_list = []
    y_list = []
    meta_list = []

    for flt in flights:
        index_csv = ROOT / f"tools/LYRA/output/F{flt}/phase1/F{flt}_frame_index.csv"
        raw_dir = ROOT / f"Data/ascope/raw/{flt}"

        if not index_csv.exists():
            print(f"  SKIP F{flt}: no frame_index.csv")
            continue
        if not raw_dir.exists():
            print(f"  SKIP F{flt}: no raw data directory")
            continue

        # Load ground truth: {tiff_id: {frame_idx: cbd_4digit}}
        gt_by_tiff = {}
        with open(index_csv) as f:
            for row in csv.DictReader(f):
                if row["frame_type"] != "complete":
                    continue
                cbd = row.get("cbd", "").strip()
                if not cbd or len(cbd) != 4:
                    continue
                tid = row["tiff_id"]
                fidx = int(row["frame_idx"])
                if tid not in gt_by_tiff:
                    gt_by_tiff[tid] = {}
                gt_by_tiff[tid][fidx] = cbd

        n_digits_flt = 0
        tiff_ids = sorted(gt_by_tiff.keys(), key=lambda x: int(x))
        for tid in tiff_ids:
            gt_cbds = gt_by_tiff[tid]
            tiff_path = _find_tiff(raw_dir, tid)
            if tiff_path is None:
                continue

            # Load TIFF
            Image.MAX_IMAGE_PIXELS = None
            img = np.array(Image.open(tiff_path), dtype=np.float32)
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
            H, W = img_norm.shape

            frames = detect_frames(img_norm)
            widths = [r - l for l, r in frames]
            med_w = float(np.median(widths))

            for fidx, (l, r) in enumerate(frames):
                if (r - l) < 0.75 * med_w:
                    continue
                cbd = gt_cbds.get(fidx)
                if not cbd:
                    continue
                # Full 7-digit string: FFF0CBD
                true_str = f"{flt:03d}{cbd}"

                crop_u8 = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
                fw = crop_u8.shape[1]
                binary = get_binary_text(crop_u8, H, fw)
                blobs = adjust_blobs_to_7(find_digit_col_ranges(binary))

                if len(blobs) != 7:
                    continue

                blob_widths = [e - s for s, e in blobs]
                med_blob_w = float(np.median(blob_widths))

                for pos, (s, e) in enumerate(blobs):
                    query, orig_aspect = crop_and_resize_digit(
                        binary, s, e, med_blob_w)
                    if query is None:
                        continue
                    if pos >= len(true_str):
                        continue

                    true_d = true_str[pos]
                    if true_d not in "0123456789":
                        continue

                    feats = extract_features(query, orig_aspect)
                    X_list.append(feats)
                    y_list.append(int(true_d))
                    meta_list.append({
                        "flight": flt,
                        "tiff_id": tid,
                        "frame_idx": fidx,
                        "pos": pos,
                    })
                    n_digits_flt += 1

        print(f"  F{flt}: {n_digits_flt} digit samples from "
              f"{len(gt_by_tiff)} TIFFs")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    print(f"  Total: {len(X)} samples, {len(set(y_list))} digit classes")
    return X, y, meta_list


# =========================================================================
# Model training and persistence
# =========================================================================

def train_model(X, y, model_path=None, test_size=0.2):
    """
    Train a RandomForestClassifier on digit features.

    Parameters
    ----------
    X : np.ndarray, shape (n, 24)
    y : np.ndarray, shape (n,) of int
    model_path : Path or None
        Where to save the model. Defaults to MODEL_PATH.
    test_size : float
        Fraction for stratified hold-out test.

    Returns
    -------
    model : trained RandomForestClassifier
    accuracy : float (test set accuracy)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    if model_path is None:
        model_path = MODEL_PATH

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    print(f"\n  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))

    print(f"\n  Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)})")
    print("\n  Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("  Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    header = "     " + "  ".join(f"{d:>3}" for d in range(10))
    print(header)
    for d in range(10):
        row = "  ".join(f"{cm[d, p]:>3}" for p in range(10))
        print(f"  {d}: {row}")

    # Feature importance
    feat_names = (
        ["seg_a", "seg_b", "seg_c", "seg_d", "seg_e", "seg_f", "seg_g"]
        + [f"hproj_{i}" for i in range(8)]
        + [f"vproj_{i}" for i in range(5)]
        + ["aspect", "density", "cx", "cy"]
    )
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n  Top 10 features:")
    for i in top_idx:
        print(f"    {feat_names[i]:>10s}: {importances[i]:.4f}")

    # Save
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, model_path)
    print(f"\n  Model saved to {model_path}")

    return model, accuracy


def load_model(model_path=None):
    """Load a trained RandomForest model from disk."""
    import joblib
    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. "
            f"Run: python tools/LYRA/ml_ocr.py --train --flights 125,126,127,128"
        )
    return joblib.load(model_path)


# =========================================================================
# Production API
# =========================================================================

def classify_digit_ml(digit_img, orig_aspect, model):
    """
    Classify a single digit image using the ML model.

    Parameters
    ----------
    digit_img : np.ndarray, float32, shape (TMPL_H, TMPL_W)
    orig_aspect : float
    model : trained sklearn classifier

    Returns
    -------
    digit : str  ('0'-'9')
    confidence : float  (probability of predicted class)
    """
    feats = extract_features(digit_img, orig_aspect).reshape(1, -1)
    pred = model.predict(feats)[0]
    proba = model.predict_proba(feats)[0]
    conf = float(proba[pred])
    return str(pred), conf


def recognize_frame_ml(frame_crop, H, fw, model):
    """
    Recognize the 7-digit CBD number from a single frame using the ML model.

    Same return signature as recognize_frame_structural() for drop-in use.

    Parameters
    ----------
    frame_crop : np.ndarray, uint8
        Cropped frame image (full height, frame width).
    H : int
        Full TIFF image height.
    fw : int
        Frame width in pixels.
    model : trained sklearn classifier

    Returns
    -------
    recognized : str or None
        7-character string (e.g. "1250465") or None if blob detection fails.
    blobs : list of (int, int)
        Column ranges of detected digit blobs.
    confidences : list of float
        Per-digit prediction probability (higher = more confident).
    """
    binary = get_binary_text(frame_crop, H, fw)
    blobs = find_digit_col_ranges(binary)

    if len(blobs) < 5:
        return None, blobs, []

    blobs = adjust_blobs_to_7(blobs)

    if len(blobs) != 7:
        return None, blobs, []

    blob_widths = [e - s for s, e in blobs]
    med_blob_w = float(np.median(blob_widths))

    recognized = ""
    confidences = []
    for s, e in blobs:
        query, orig_aspect = crop_and_resize_digit(binary, s, e, med_blob_w)
        if query is None:
            recognized += "?"
            confidences.append(0.0)
            continue
        digit, conf = classify_digit_ml(query, orig_aspect, model)
        recognized += digit
        confidences.append(conf)

    return recognized, blobs, confidences


# =========================================================================
# Ensemble: segment OCR + ML fusion
# =========================================================================

def ensemble_select_best(complete_indices, seg_reads, ml_reads, flight):
    """
    Run both segment and ML read lists through the sequential constraint
    independently, then pick whichever set produces more anchors.

    When both agree on a frame, that frame gets confidence 1.0.
    When they disagree, the winning method's frames get 0.6.

    Parameters
    ----------
    complete_indices : list of int
        Frame indices for complete frames.
    seg_reads : list of str or None
        Segment OCR raw reads (one per complete frame).
    ml_reads : list of str or None
        ML raw reads (one per complete frame).
    flight : int
        Flight number.

    Returns
    -------
    best_reads : list of str or None
        The raw reads from the winning method.
    confidences : list of float
        Per-frame confidence (1.0 = both agree, 0.6 = winner only).
    method_label : str
        "ensemble(seg)" or "ensemble(ml)" or "ensemble(agree)".
    """
    from build_digit_templates import apply_sequential_constraint

    _, seg_anchors = apply_sequential_constraint(
        complete_indices, seg_reads, flight=flight)
    _, ml_anchors = apply_sequential_constraint(
        complete_indices, ml_reads, flight=flight)

    # Pick the method with more anchors
    if seg_anchors > ml_anchors:
        best_reads = seg_reads
        method_label = "ensemble(seg)"
    elif ml_anchors > seg_anchors:
        best_reads = ml_reads
        method_label = "ensemble(ml)"
    else:
        # Tie — prefer ML (better calibrated confidence)
        best_reads = ml_reads
        method_label = "ensemble(ml)"

    # Compute per-frame confidence based on agreement
    confidences = []
    for seg_r, ml_r in zip(seg_reads, ml_reads):
        if seg_r and ml_r and seg_r == ml_r:
            confidences.append(1.0)
        elif seg_r or ml_r:
            confidences.append(0.6)
        else:
            confidences.append(0.0)

    return best_reads, confidences, method_label


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML digit classifier for LYRA CBD recognition")
    parser.add_argument("--train", action="store_true",
                        help="Train model on labeled data")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model (same as train but no save)")
    parser.add_argument("--predict", type=str, default=None,
                        help="Predict CBDs for a single TIFF file")
    parser.add_argument("--flights", type=str, default="125,126,127,128",
                        help="Comma-separated flight numbers for training")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (default: models/digit_rf.pkl)")
    args = parser.parse_args()

    flights = [int(f) for f in args.flights.split(",")]
    model_path = Path(args.model) if args.model else MODEL_PATH

    if args.train or args.evaluate:
        print(f"Loading training data from flights: {flights}")
        X, y, meta = load_training_data(flights)

        if len(X) == 0:
            print("ERROR: No training data found.")
            sys.exit(1)

        if args.train:
            train_model(X, y, model_path)
        else:
            # Evaluate only — don't save
            train_model(X, y, model_path=None, test_size=0.2)

    elif args.predict:
        tiff_path = Path(args.predict)
        if not tiff_path.exists():
            sys.exit(f"ERROR: TIFF not found: {tiff_path}")

        model = load_model(model_path)

        Image.MAX_IMAGE_PIXELS = None
        img = np.array(Image.open(tiff_path), dtype=np.float32)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
        H, W = img_norm.shape

        frames = detect_frames(img_norm)
        widths = [r - l for l, r in frames]
        med_w = float(np.median(widths))

        print(f"TIFF: {tiff_path.name}")
        print(f"Frames detected: {len(frames)}")
        print(f"{'Frame':>6s}  {'ML pred':>8s}  {'Conf':>6s}")
        print("-" * 28)

        for fidx, (l, r) in enumerate(frames):
            if (r - l) < 0.75 * med_w:
                print(f"  {fidx:>4d}  {'(partial)':>8s}")
                continue
            crop_u8 = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
            fw = crop_u8.shape[1]
            recog, blobs, confs = recognize_frame_ml(crop_u8, H, fw, model)
            if recog:
                mean_conf = np.mean(confs) if confs else 0.0
                cbd = recog[3:] if len(recog) >= 7 else recog
                print(f"  {fidx:>4d}  {recog:>8s}  {mean_conf:>5.1%}")
            else:
                print(f"  {fidx:>4d}  {'(fail)':>8s}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
