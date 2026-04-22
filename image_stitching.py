"""
Image Stitching Assignment - CV2026
Implements panorama stitching using Harris corners, patch descriptors,
RANSAC-based affine estimation, and image warping.

Usage:
    python image_stitching.py --img1 left.jpg --img2 right.jpg [options]
    python image_stitching.py --demo   # runs on synthetic test pair

Options:
    --img1        Path to left image
    --img2        Path to right image
    --patch_size  Descriptor patch size (default: 21)
    --n_matches   Number of top matches to keep (default: 200)
    --ransac_iter Number of RANSAC iterations (default: 1000)
    --ransac_thr  RANSAC inlier threshold in pixels (default: 5.0)
    --metric      Matching metric: 'ncc' or 'euclidean' (default: 'euclidean')
    --sensitivity Run sensitivity analysis
    --output_dir  Directory to save results (default: ./results)
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist

# ──────────────────────────────────────────────────────────────────────────────
# 1. FEATURE DETECTION  (Harris corners)
# ──────────────────────────────────────────────────────────────────────────────

def detect_harris_keypoints(img_gray, max_keypoints=500, min_distance=10, sigma=1.0):
    """
    Detect Harris corner keypoints using skimage.
    Returns array of (row, col) coordinates.
    """
    response = corner_harris(img_gray, sigma=sigma)
    coords = corner_peaks(response, min_distance=min_distance,
                          num_peaks=max_keypoints, threshold_rel=0.01)
    return coords  # shape (N, 2): each row is (row, col)


# ──────────────────────────────────────────────────────────────────────────────
# 2. DESCRIPTOR EXTRACTION  (fixed-size patch around each keypoint)
# ──────────────────────────────────────────────────────────────────────────────

def extract_patch_descriptors(img_gray, keypoints, patch_size=21):
    """
    Extract a flattened, normalised intensity patch around every keypoint.
    Keypoints too close to the border are discarded.

    Returns:
        valid_kps   – filtered keypoints (M, 2)
        descriptors – (M, patch_size^2) float32 array
    """
    half = patch_size // 2
    h, w = img_gray.shape
    valid_kps, descs = [], []

    for r, c in keypoints:
        if r - half < 0 or r + half + 1 > h or c - half < 0 or c + half + 1 > w:
            continue
        patch = img_gray[r - half:r + half + 1, c - half:c + half + 1].copy()
        patch = patch.astype(np.float32)
        # Zero-mean, unit-variance normalisation
        std = patch.std()
        if std < 1e-6:
            continue
        patch = (patch - patch.mean()) / std
        valid_kps.append((r, c))
        descs.append(patch.flatten())

    return np.array(valid_kps), np.array(descs, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3. DESCRIPTOR MATCHING
# ──────────────────────────────────────────────────────────────────────────────

def match_descriptors_ncc(descs1, descs2, n_matches=200):
    """
    Match using Normalised Cross-Correlation (higher = better).
    Each descriptor is already zero-mean/unit-var, so NCC = dot product / N.
    """
    # (M1, D) x (D, M2) -> (M1, M2)  correlation matrix
    ncc = descs1 @ descs2.T / descs1.shape[1]
    # For each descriptor in img1, find the best match in img2
    best_idx = np.argmax(ncc, axis=1)
    scores = ncc[np.arange(len(descs1)), best_idx]
    order = np.argsort(scores)[::-1][:n_matches]
    matches = [(i, best_idx[i], scores[i]) for i in order]
    return matches  # list of (idx1, idx2, score)


def match_descriptors_euclidean(descs1, descs2, n_matches=200):
    """
    Match using Euclidean distance after L2-normalising each descriptor.
    Lower = better; we return as (idx1, idx2, -distance) so higher score = better.
    """
    # L2-normalise
    d1 = descs1 / (np.linalg.norm(descs1, axis=1, keepdims=True) + 1e-8)
    d2 = descs2 / (np.linalg.norm(descs2, axis=1, keepdims=True) + 1e-8)
    dists = cdist(d1, d2, metric="euclidean")
    best_idx = np.argmin(dists, axis=1)
    scores = -dists[np.arange(len(d1)), best_idx]  # negate so higher = better
    order = np.argsort(scores)[::-1][:n_matches]
    matches = [(i, best_idx[i], scores[i]) for i in order]
    return matches


def match_descriptors(descs1, descs2, n_matches=200, metric="euclidean"):
    if metric == "ncc":
        return match_descriptors_ncc(descs1, descs2, n_matches)
    return match_descriptors_euclidean(descs1, descs2, n_matches)


# ──────────────────────────────────────────────────────────────────────────────
# 4. RANSAC – Affine Transformation Estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_affine_lstsq(src_pts, dst_pts):
    """
    Fit a 2D affine transform T such that dst ≈ T @ [src; 1].
    Uses least-squares (numpy.linalg.lstsq).

    Returns 3×3 affine matrix (homogeneous form).
    """
    n = len(src_pts)
    # Build system A x = b  for both x and y coordinates simultaneously
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(src_pts, dst_pts)):
        A[2 * i]     = [x1, y1, 1, 0,  0,  0]
        A[2 * i + 1] = [0,  0,  0, x1, y1, 1]
        b[2 * i]     = x2
        b[2 * i + 1] = y2
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a0, a1, a2, a3, a4, a5 = params
    M = np.array([[a0, a1, a2],
                  [a3, a4, a5],
                  [0,  0,  1 ]], dtype=np.float64)
    return M


def apply_affine(M, pts):
    """Apply 3×3 affine matrix to Nx2 array of (x,y) points."""
    ones = np.ones((len(pts), 1))
    homo = np.hstack([pts, ones])          # (N, 3)
    transformed = (M @ homo.T).T          # (N, 3)
    return transformed[:, :2]


def ransac_affine(kps1, kps2, matches, n_iter=1000, threshold=5.0, n_sample=4):
    """
    RANSAC to robustly estimate the affine transform mapping kps2 → kps1.

    kps1, kps2 : (N,2) arrays in (row, col) → we treat as (y,x),
                  but for affine fitting we use (col, row) = (x, y).
    matches     : list of (idx1, idx2, score)
    Returns best_M, inlier_mask (bool array over matches)
    """
    if len(matches) < n_sample:
        return None, np.zeros(len(matches), dtype=bool)

    # Convert keypoints to (x,y) convention
    pts1 = np.array([(kps1[m[0]][1], kps1[m[0]][0]) for m in matches], dtype=np.float64)
    pts2 = np.array([(kps2[m[1]][1], kps2[m[1]][0]) for m in matches], dtype=np.float64)

    best_M = None
    best_inliers = np.zeros(len(matches), dtype=bool)
    best_n_inliers = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(len(matches), n_sample, replace=False)
        try:
            M = estimate_affine_lstsq(pts2[idx], pts1[idx])
        except Exception:
            continue

        # Evaluate: transform all pts2, measure distance to pts1
        projected = apply_affine(M, pts2)
        residuals = np.linalg.norm(projected - pts1, axis=1)
        inliers = residuals < threshold

        n_in = inliers.sum()
        if n_in > best_n_inliers:
            best_n_inliers = n_in
            best_inliers = inliers.copy()
            best_M = M

    # Refit on all inliers for a better estimate
    if best_n_inliers >= n_sample:
        try:
            best_M = estimate_affine_lstsq(pts2[best_inliers], pts1[best_inliers])
        except Exception:
            pass

    return best_M, best_inliers


# ──────────────────────────────────────────────────────────────────────────────
# 5. IMAGE WARPING
# ──────────────────────────────────────────────────────────────────────────────

def warp_and_stitch(img1, img2, M):
    """
    Warp img2 onto img1's coordinate frame using affine M (3×3),
    then composite them into a single panorama.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Map the four corners of img2 into img1's frame to find canvas size
    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float64)
    corners2_transformed = apply_affine(M, corners2)

    all_corners = np.vstack([
        [[0, 0], [w1, 0], [w1, h1], [0, h1]],
        corners2_transformed
    ])
    x_min, y_min = all_corners.min(axis=0)
    x_max, y_max = all_corners.max(axis=0)

    tx = int(-x_min) if x_min < 0 else 0
    ty = int(-y_min) if y_min < 0 else 0
    canvas_w = int(x_max - x_min) + 1
    canvas_h = int(y_max - y_min) + 1

    # Translation matrix to shift everything into positive canvas coords
    T_shift = np.array([[1, 0, tx],
                        [0, 1, ty],
                        [0, 0, 1 ]], dtype=np.float64)
    M_shifted = T_shift @ M

    # Warp img2
    warped2 = cv2.warpAffine(img2, M_shifted[:2], (canvas_w, canvas_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Place img1 on canvas
    canvas = warped2.copy()
    canvas[ty:ty + h1, tx:tx + w1] = img1

    return canvas, tx, ty


# ──────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def draw_keypoints(img, kps, color=(0, 255, 0), radius=3):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for r, c in kps:
        cv2.circle(out, (int(c), int(r)), radius, color, -1)
    return out


def draw_matches(img1, kps1, img2, kps2, matches, inlier_mask=None, max_draw=100):
    """Draw matching lines between two images side by side."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:] = img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for i, (i1, i2, _) in enumerate(matches[:max_draw]):
        if inlier_mask is not None:
            color = (0, 200, 0) if inlier_mask[i] else (0, 0, 180)
        else:
            color = (200, 200, 0)
        r1, c1 = kps1[i1]
        r2, c2 = kps2[i2]
        pt1 = (int(c1), int(r1))
        pt2 = (int(c2) + w1, int(r2))
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, (255, 255, 0), -1)
        cv2.circle(canvas, pt2, 3, (255, 255, 0), -1)
    return canvas


def save_figure(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. SENSITIVITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def score_alignment(kps1, kps2, matches, inlier_mask, M):
    """
    Score = mean Euclidean distance between transformed kps2 inliers and kps1 inliers.
    Lower is better.
    """
    if M is None or inlier_mask.sum() == 0:
        return np.inf
    inlier_matches = [m for m, flag in zip(matches, inlier_mask) if flag]
    pts1 = np.array([(kps1[m[0]][1], kps1[m[0]][0]) for m in inlier_matches])
    pts2 = np.array([(kps2[m[1]][1], kps2[m[1]][0]) for m in inlier_matches])
    projected = apply_affine(M, pts2)
    return float(np.mean(np.linalg.norm(projected - pts1, axis=1)))


def sensitivity_analysis(img1_gray, img2_gray, img1_bgr, img2_bgr, out_dir):
    """Run sensitivity sweeps over patch_size, n_matches, ransac_threshold."""
    print("\n[Sensitivity Analysis]")
    os.makedirs(out_dir, exist_ok=True)

    kps1_raw = detect_harris_keypoints(img1_gray, max_keypoints=800)
    kps2_raw = detect_harris_keypoints(img2_gray, max_keypoints=800)

    # ── a) Patch size sweep ──────────────────────────────────────────────────
    patch_sizes = [11, 15, 21, 31, 41]
    scores_patch = []
    print("  Sweeping patch_size:", patch_sizes)
    for ps in patch_sizes:
        kp1, d1 = extract_patch_descriptors(img1_gray, kps1_raw, ps)
        kp2, d2 = extract_patch_descriptors(img2_gray, kps2_raw, ps)
        if len(d1) < 10 or len(d2) < 10:
            scores_patch.append(np.inf); continue
        matches = match_descriptors(d1, d2, n_matches=200)
        M, mask = ransac_affine(kp1, kp2, matches, n_iter=500, threshold=5.0)
        scores_patch.append(score_alignment(kp1, kp2, matches, mask, M))
    print("  Scores:", [f"{s:.3f}" for s in scores_patch])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(patch_sizes, scores_patch, "o-", color="#2563eb", linewidth=2)
    ax.set_xlabel("Patch size (pixels)"); ax.set_ylabel("Mean inlier distance (px)")
    ax.set_title("Sensitivity: Patch Size"); ax.grid(True, alpha=0.3)
    save_figure(fig, os.path.join(out_dir, "sensitivity_patch_size.png"))

    # ── b) Number of matches sweep ───────────────────────────────────────────
    kp1, d1 = extract_patch_descriptors(img1_gray, kps1_raw, 21)
    kp2, d2 = extract_patch_descriptors(img2_gray, kps2_raw, 21)
    n_match_vals = [50, 100, 150, 200, 300, 400]
    scores_nm, inlier_ratios = [], []
    print("  Sweeping n_matches:", n_match_vals)
    for nm in n_match_vals:
        nm = min(nm, min(len(d1), len(d2)))
        matches = match_descriptors(d1, d2, n_matches=nm)
        M, mask = ransac_affine(kp1, kp2, matches, n_iter=500, threshold=5.0)
        scores_nm.append(score_alignment(kp1, kp2, matches, mask, M))
        inlier_ratios.append(mask.sum() / len(mask) if len(mask) > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(n_match_vals, scores_nm, "s-", color="#16a34a", linewidth=2)
    axes[0].set_xlabel("# matches kept"); axes[0].set_ylabel("Mean inlier distance (px)")
    axes[0].set_title("Sensitivity: # Matches (Score)"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(n_match_vals, inlier_ratios, "^-", color="#dc2626", linewidth=2)
    axes[1].set_xlabel("# matches kept"); axes[1].set_ylabel("Inlier ratio")
    axes[1].set_title("Sensitivity: # Matches (Inlier ratio)"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "sensitivity_n_matches.png"))

    # ── c) RANSAC threshold sweep ────────────────────────────────────────────
    matches = match_descriptors(d1, d2, n_matches=200)
    thresholds = [2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    scores_thr, inlier_ratios_thr = [], []
    print("  Sweeping RANSAC threshold:", thresholds)
    for thr in thresholds:
        M, mask = ransac_affine(kp1, kp2, matches, n_iter=500, threshold=thr)
        scores_thr.append(score_alignment(kp1, kp2, matches, mask, M))
        inlier_ratios_thr.append(mask.sum() / len(mask) if len(mask) > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(thresholds, scores_thr, "o-", color="#7c3aed", linewidth=2)
    axes[0].set_xlabel("RANSAC threshold (px)"); axes[0].set_ylabel("Mean inlier distance (px)")
    axes[0].set_title("Sensitivity: RANSAC Threshold (Score)"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(thresholds, inlier_ratios_thr, "^-", color="#ea580c", linewidth=2)
    axes[1].set_xlabel("RANSAC threshold (px)"); axes[1].set_ylabel("Inlier ratio")
    axes[1].set_title("Sensitivity: RANSAC Threshold (Inlier ratio)"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "sensitivity_ransac_threshold.png"))

    # ── d) NCC vs Euclidean comparison ──────────────────────────────────────
    results = {}
    for metric in ["ncc", "euclidean"]:
        matches = match_descriptors(d1, d2, n_matches=200, metric=metric)
        M, mask = ransac_affine(kp1, kp2, matches, n_iter=500, threshold=5.0)
        results[metric] = {
            "score": score_alignment(kp1, kp2, matches, mask, M),
            "inliers": int(mask.sum()),
            "outliers": int((~mask).sum()),
        }
    print("  NCC result:", results["ncc"])
    print("  Euclidean result:", results["euclidean"])

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = list(results.keys())
    scores = [results[m]["score"] for m in metrics]
    inliers = [results[m]["inliers"] for m in metrics]
    x = np.arange(len(metrics))
    bars = ax.bar(x - 0.2, scores, 0.35, label="Score (px)", color="#2563eb")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, inliers, 0.35, label="# Inliers", color="#16a34a", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["NCC", "Euclidean"])
    ax.set_ylabel("Mean inlier distance (px)"); ax2.set_ylabel("# Inliers")
    ax.set_title("NCC vs Euclidean Matching")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    save_figure(fig, os.path.join(out_dir, "sensitivity_ncc_vs_euclidean.png"))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(img1_bgr, img2_bgr, patch_size=21, n_matches=200,
                 ransac_iter=1000, ransac_thr=5.0, metric="euclidean",
                 out_dir="results", prefix="pair"):
    """Full stitching pipeline for one image pair. Saves all intermediate figures."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Running pipeline | prefix={prefix} | patch={patch_size} | "
          f"matches={n_matches} | ransac_iter={ransac_iter} | thr={ransac_thr} | metric={metric}")
    print(f"{'='*60}")

    img1_gray = rgb2gray(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)
    img2_gray = rgb2gray(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)

    # ── Step 1: Harris keypoints ─────────────────────────────────────────────
    print("[1] Detecting Harris keypoints …")
    kps1_raw = detect_harris_keypoints(img1_gray, max_keypoints=800)
    kps2_raw = detect_harris_keypoints(img2_gray, max_keypoints=800)
    print(f"    Detected {len(kps1_raw)} / {len(kps2_raw)} keypoints")

    # Visualise
    kp_vis1 = draw_keypoints(img1_bgr, kps1_raw)
    kp_vis2 = draw_keypoints(img2_bgr, kps2_raw)
    kp_vis = np.hstack([kp_vis1, kp_vis2])
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_01_keypoints.png"), kp_vis)
    print(f"  Saved: {prefix}_01_keypoints.png")

    # ── Step 2: Descriptors ──────────────────────────────────────────────────
    print("[2] Extracting patch descriptors …")
    kps1, descs1 = extract_patch_descriptors(img1_gray, kps1_raw, patch_size)
    kps2, descs2 = extract_patch_descriptors(img2_gray, kps2_raw, patch_size)
    print(f"    Valid: {len(kps1)} / {len(kps2)} after border filtering")

    if len(descs1) < 10 or len(descs2) < 10:
        print("  ERROR: Too few descriptors. Aborting.")
        return None

    # ── Step 3: Matching ─────────────────────────────────────────────────────
    print(f"[3] Matching descriptors (metric={metric}, top {n_matches}) …")
    n_matches = min(n_matches, len(descs1), len(descs2))
    matches = match_descriptors(descs1, descs2, n_matches=n_matches, metric=metric)

    # ── Step 4: RANSAC ───────────────────────────────────────────────────────
    print(f"[4] RANSAC ({ransac_iter} iters, thr={ransac_thr}px) …")
    t0 = time.time()
    M, inlier_mask = ransac_affine(kps1, kps2, matches,
                                   n_iter=ransac_iter, threshold=ransac_thr)
    elapsed = time.time() - t0
    n_in = int(inlier_mask.sum())
    n_out = int((~inlier_mask).sum())
    print(f"    Inliers: {n_in}  Outliers: {n_out}  ({elapsed:.2f}s)")

    if M is None:
        print("  ERROR: RANSAC failed. Aborting.")
        return None

    # Residuals for inliers
    inlier_matches = [m for m, f in zip(matches, inlier_mask) if f]
    pts1_in = np.array([(kps1[m[0]][1], kps1[m[0]][0]) for m in inlier_matches])
    pts2_in = np.array([(kps2[m[1]][1], kps2[m[1]][0]) for m in inlier_matches])
    proj = apply_affine(M, pts2_in)
    residuals = np.linalg.norm(proj - pts1_in, axis=1)
    avg_residual = float(np.mean(residuals))
    print(f"    Average inlier residual: {avg_residual:.3f} px")
    print(f"    Estimated transform:\n{M}")

    # Visualise matches
    match_vis = draw_matches(img1_bgr, kps1, img2_bgr, kps2, matches,
                             inlier_mask=inlier_mask, max_draw=150)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_02_matches.png"), match_vis)
    print(f"  Saved: {prefix}_02_matches.png")

    # Visualise inliers only
    inlier_vis = draw_matches(img1_bgr, kps1, img2_bgr, kps2,
                              [m for m, f in zip(matches, inlier_mask) if f],
                              max_draw=100)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_03_inliers.png"), inlier_vis)
    print(f"  Saved: {prefix}_03_inliers.png")

    # ── Step 5: Warp & stitch ────────────────────────────────────────────────
    print("[5] Warping and stitching …")
    panorama, tx, ty = warp_and_stitch(img1_bgr, img2_bgr, M)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_04_panorama.png"), panorama)
    print(f"  Saved: {prefix}_04_panorama.png")

    # ── Summary figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].imshow(cv2.cvtColor(kp_vis, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Harris Keypoints  ({len(kps1)} | {len(kps2)})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"All Matches  ({len(matches)} shown)")
    green_patch = mpatches.Patch(color="green", label=f"Inliers ({n_in})")
    red_patch   = mpatches.Patch(color="red",   label=f"Outliers ({n_out})")
    axes[0, 1].legend(handles=[green_patch, red_patch], fontsize=8)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(inlier_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Inlier Matches  (avg residual: {avg_residual:.2f} px)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Stitched Panorama")
    axes[1, 1].axis("off")

    fig.suptitle(f"Image Stitching Summary | {prefix}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"{prefix}_05_summary.png"))

    return {
        "M": M,
        "n_inliers": n_in,
        "n_outliers": n_out,
        "avg_residual": avg_residual,
        "inlier_ratio": n_in / (n_in + n_out) if (n_in + n_out) > 0 else 0,
        "score": avg_residual,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9. DEMO MODE  (synthetic test images so the script is self-contained)
# ──────────────────────────────────────────────────────────────────────────────

def make_demo_pair():
    """
    Create a synthetic overlapping pair:
      - img1: a random textured image
      - img2: img1 shifted by (120px, 10px) with a slight rotation
    """
    np.random.seed(0)
    h, w = 400, 600

    # Random checkerboard + gradient texture
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, h, 40):
        for j in range(0, w, 40):
            color = np.random.randint(50, 220, 3).tolist()
            base[i:i+40, j:j+40] = color
    # Add some noise
    noise = np.random.randint(0, 30, base.shape, dtype=np.uint8)
    base = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    # Add gradient
    grad = np.linspace(0, 60, w, dtype=np.uint8)[np.newaxis, :, np.newaxis]
    base = np.clip(base.astype(int) + grad, 0, 255).astype(np.uint8)

    img1 = base.copy()

    # Affine: shift + tiny rotation
    dx, dy, angle = 120, 8, 1.5  # degrees
    cx, cy = w / 2, h / 2
    rad = np.deg2rad(angle)
    M_true = np.array([
        [np.cos(rad), -np.sin(rad), dx + cx * (1 - np.cos(rad)) + cy * np.sin(rad)],
        [np.sin(rad),  np.cos(rad), dy + cy * (1 - np.cos(rad)) - cx * np.sin(rad)],
        [0,            0,           1]
    ])
    # img2 is img1 warped by M_true; we crop a window to simulate "right image"
    warped = cv2.warpAffine(img1, M_true[:2], (w + dx + 50, h + 50))
    img2 = warped[:h, dx - 20: dx - 20 + w]
    if img2.shape[1] < 100:
        img2 = warped[:h, :w]

    return img1, img2


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image Stitching – CV2026 Assignment")
    parser.add_argument("--img1", default=None)
    parser.add_argument("--img2", default=None)
    parser.add_argument("--patch_size", type=int, default=21)
    parser.add_argument("--n_matches", type=int, default=200)
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--ransac_thr", type=float, default=5.0)
    parser.add_argument("--metric", choices=["ncc", "euclidean"], default="euclidean")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    if args.demo or (args.img1 is None and args.img2 is None):
        print("[DEMO MODE] Creating synthetic image pair …")
        img1, img2 = make_demo_pair()
        prefix = "demo"
    else:
        if not os.path.exists(args.img1) or not os.path.exists(args.img2):
            print(f"ERROR: Cannot find images: {args.img1}, {args.img2}")
            sys.exit(1)
        img1 = cv2.imread(args.img1)
        img2 = cv2.imread(args.img2)
        if img1 is None or img2 is None:
            print("ERROR: Failed to read images.")
            sys.exit(1)
        prefix = "pair"

    result = run_pipeline(
        img1, img2,
        patch_size=args.patch_size,
        n_matches=args.n_matches,
        ransac_iter=args.ransac_iter,
        ransac_thr=args.ransac_thr,
        metric=args.metric,
        out_dir=args.output_dir,
        prefix=prefix,
    )

    if result:
        print("\n── RESULTS ─────────────────────────────────────")
        print(f"  Inliers         : {result['n_inliers']}")
        print(f"  Outliers        : {result['n_outliers']}")
        print(f"  Inlier ratio    : {result['inlier_ratio']:.3f}")
        print(f"  Avg residual    : {result['avg_residual']:.3f} px")
        print(f"  Affine matrix   :\n{result['M']}")
        print(f"  Output saved to : {args.output_dir}/")

    if args.sensitivity:
        img1_gray = rgb2gray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).astype(np.float32)
        img2_gray = rgb2gray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).astype(np.float32)
        sensitivity_analysis(img1_gray, img2_gray, img1, img2,
                             out_dir=os.path.join(args.output_dir, "sensitivity"))


if __name__ == "__main__":
    main()
