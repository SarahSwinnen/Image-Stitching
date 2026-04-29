#import libraries
from skimage.feature import corner_harris, corner_peaks
import matplotlib.pyplot as plt
import numpy as np
import cv2


def makeGray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    return gray


def harrisCornerDetection(imgGray, min_distance=4, num_peaks=500, threshold_rel=0.001):
    response = corner_harris(imgGray)
    coords = corner_peaks(
        response,
        min_distance=min_distance,
        num_peaks=num_peaks,
        threshold_rel=threshold_rel,
    )

    keypoints = np.array([[c[1], c[0]] for c in coords], dtype=np.float32)
    return keypoints, response


def patches(imgGray, keypoints, patchSize=11):
    if patchSize % 2 == 0:
        raise ValueError("patchSize must be odd so the patch is centered on the keypoint")

    half = patchSize // 2
    h, w = imgGray.shape
    valid_kps, descs = [], []

    for x, y in keypoints:
        x = int(round(x))
        y = int(round(y))

        if y - half < 0 or y + half + 1 > h or x - half < 0 or x + half + 1 > w:
            continue

        patch = imgGray[y - half:y + half + 1, x - half:x + half + 1].copy().astype(np.float32)

        std = patch.std()
        if std < 1e-6:
            continue

        patch = (patch - patch.mean()) / std
        valid_kps.append((x, y))
        descs.append(patch.flatten())

    if len(descs) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, patchSize * patchSize), dtype=np.float32)

    return np.array(valid_kps, dtype=np.float32), np.array(descs, dtype=np.float32)


def sift(imgGray, keypoints):
    size = 16

    if imgGray.dtype != np.uint8:
        if imgGray.max() <= 1.0:
            imgGray = (imgGray * 255).astype(np.uint8)
        else:
            imgGray = np.clip(imgGray, 0, 255).astype(np.uint8)

    sift_detector = cv2.SIFT_create()

    cv2_keypoints = [
        cv2.KeyPoint(float(x), float(y), size)
        for x, y in keypoints
    ]

    cv2_keypoints, descriptors = sift_detector.compute(imgGray, cv2_keypoints)

    if descriptors is None or len(cv2_keypoints) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    final_keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in cv2_keypoints], dtype=np.float32)
    return final_keypoints, descriptors.astype(np.float32)


def normalize_descriptors(desc):
    if len(desc) == 0:
        return desc
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return desc / norms


def normCorr(d1, d2):
    return d1 @ d2.T


def euclidean(d1, d2):
    d1sqr = np.sum(d1 ** 2, axis=1, keepdims=True)
    d2sqr = np.sum(d2 ** 2, axis=1, keepdims=True).T
    distances = d1sqr + d2sqr - 2 * (d1 @ d2.T)
    distances = np.maximum(distances, 0.0)
    return np.sqrt(distances)


def bestMatchesCorr(d1, d2, numMatches=200):
    if len(d1) == 0 or len(d2) == 0:
        return []

    d1_norm = normalize_descriptors(d1)
    d2_norm = normalize_descriptors(d2)
    corr = normCorr(d1_norm, d2_norm)
    best_idx = np.argmax(corr, axis=1)
    scores = corr[np.arange(len(d1_norm)), best_idx]
    order = np.argsort(scores)[::-1][:numMatches]
    matches = [(i, best_idx[i], scores[i]) for i in order]
    return matches


def bestMatchesEuc(d1, d2, ratio=0.75, numMatches=200):
    if len(d1) == 0 or len(d2) == 0:
        return []

    d1_norm = normalize_descriptors(d1)
    d2_norm = normalize_descriptors(d2)
    distance = euclidean(d1_norm, d2_norm)
    matches = []

    for i in range(distance.shape[0]):
        sorted_idx = np.argsort(distance[i])
        if len(sorted_idx) < 2:
            continue

        j1, j2 = sorted_idx[0], sorted_idx[1]
        d_first, d_second = distance[i, j1], distance[i, j2]

        if d_first / (d_second + 1e-8) < ratio:
            matches.append((i, j1, d_first))

    matches.sort(key=lambda x: x[2])
    return matches[:numMatches]


def affineLeastSquares(sourcePoints, destinationPoints):
    """
    x' = a*x + b*y + tx
    y' = c*x + d*y + ty
    """
    n = sourcePoints.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to estimate affine transform")

    A = []
    B = []

    for i in range(n):
        x, y = sourcePoints[i]
        xp, yp = destinationPoints[i]

        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])

        B.append(xp)
        B.append(yp)

    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)

    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    M = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0, 0, 1]
    ], dtype=np.float32)

    return M


def applyAffine(M, pts):
    ones = np.ones((len(pts), 1), dtype=np.float32)
    homo = np.hstack([pts, ones])
    transformed = (M @ homo.T).T
    return transformed[:, :2]


def ransac_affine(kps1, kps2, matches, n_iter=1000, threshold=3.0, n_sample=4, random_seed=42):
    if len(matches) < n_sample:
        return None, np.zeros(len(matches), dtype=bool), None, None

    pts1 = np.array([kps1[m[0]] for m in matches], dtype=np.float64)
    pts2 = np.array([kps2[m[1]] for m in matches], dtype=np.float64)

    best_M = None
    best_inliers = np.zeros(len(matches), dtype=bool)
    best_n_inliers = 0
    best_avg_residual = np.inf

    rng = np.random.default_rng(random_seed)

    for _ in range(n_iter):
        idx = rng.choice(len(matches), n_sample, replace=False)
        try:
            M = affineLeastSquares(pts2[idx], pts1[idx])
        except Exception:
            continue

        projected = applyAffine(M, pts2)
        residuals = np.linalg.norm(projected - pts1, axis=1)
        inliers = residuals < threshold
        n_in = int(inliers.sum())
        avg_residual = float(np.mean(residuals[inliers])) if n_in > 0 else np.inf

        better_model = (
            (n_in > best_n_inliers) or
            (n_in == best_n_inliers and avg_residual < best_avg_residual)
        )

        if better_model:
            best_n_inliers = n_in
            best_avg_residual = avg_residual
            best_inliers = inliers.copy()
            best_M = M

    if best_n_inliers >= n_sample:
        try:
            best_M = affineLeastSquares(pts2[best_inliers], pts1[best_inliers])
        except Exception:
            pass

    return best_M, best_inliers, pts1, pts2


def transformPoints(M, pts):
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    transformed = (M @ pts_h.T).T
    return transformed[:, :2]


def warpStitch(img1, img2, M):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.array([
        [0, 0],
        [w2 - 1, 0],
        [w2 - 1, h2 - 1],
        [0, h2 - 1]
    ], dtype=np.float32)

    corners1 = np.array([
        [0, 0],
        [w1 - 1, 0],
        [w1 - 1, h1 - 1],
        [0, h1 - 1]
    ], dtype=np.float32)

    warped_corners2 = transformPoints(M, corners2)
    all_corners = np.vstack([corners1, warped_corners2])

    min_x = np.floor(np.min(all_corners[:, 0])).astype(int)
    min_y = np.floor(np.min(all_corners[:, 1])).astype(int)
    max_x = np.ceil(np.max(all_corners[:, 0])).astype(int)
    max_y = np.ceil(np.max(all_corners[:, 1])).astype(int)

    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1

    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    M_total = translation @ M
    warped_img2 = cv2.warpPerspective(img2, M_total, (out_w, out_h))

    panorama = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    panorama[:] = warped_img2

    x_offset = -min_x
    y_offset = -min_y

    roi = panorama[y_offset:y_offset + h1, x_offset:x_offset + w1]
    mask_img1 = np.any(img1 > 0, axis=2)
    mask_roi = np.any(roi > 0, axis=2)

    both = mask_img1 & mask_roi
    only_img1 = mask_img1 & (~mask_roi)

    roi[only_img1] = img1[only_img1]
    roi[both] = ((roi[both].astype(np.float32) + img1[both].astype(np.float32)) / 2).astype(np.uint8)
    panorama[y_offset:y_offset + h1, x_offset:x_offset + w1] = roi

    return panorama


def affineError(M, kp1, kp2, matches):
    if len(matches) == 0:
        return np.array([], dtype=np.float32)

    pts1 = np.array([kp1[m[0]] for m in matches], dtype=np.float64)
    pts2 = np.array([kp2[m[1]] for m in matches], dtype=np.float64)
    pts2_transformed = applyAffine(M, pts2)
    errors = np.linalg.norm(pts1 - pts2_transformed, axis=1)
    return errors


def compute_accuracy_score(M, kp1, kp2, matches, use_inliers_only=False, inlier_mask=None):
    selected_matches = list(matches)
    if use_inliers_only:
        if inlier_mask is None:
            raise ValueError("inlier_mask is required when use_inliers_only=True")
        selected_matches = [m for m, f in zip(matches, inlier_mask) if f]

    errors = affineError(M, kp1, kp2, selected_matches)
    if len(errors) == 0:
        return np.inf
    return float(np.mean(errors))


def load_images(image1_path, image2_path):
    image1_bgr = cv2.imread(image1_path)
    image2_bgr = cv2.imread(image2_path)

    if image1_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image1_path}")
    if image2_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image2_path}")

    image1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)
    gray1 = makeGray(image1)
    gray2 = makeGray(image2)
    return image1, image2, gray1, gray2


def pipeline(
    image1,
    image2,
    gray1,
    gray2,
    min_distance=4,
    num_peaks=500,
    threshold_rel=0.001,
    useSift=False,
    patchSize=11,
    matchChoice="euclidean",
    ratio=0.75,
    numMatches=200,
    ransac_iter=1000,
    ransac_threshold=3.0,
    ransac_sample_size=4,
    show_panorama=False,
    print_summary=True,
):
    keypoints1, harrisResponse1 = harrisCornerDetection(gray1, min_distance, num_peaks, threshold_rel)
    keypoints2, harrisResponse2 = harrisCornerDetection(gray2, min_distance, num_peaks, threshold_rel)

    if useSift:
        kp1desc, desc1 = sift(gray1, keypoints1)
        kp2desc, desc2 = sift(gray2, keypoints2)
    else:
        kp1desc, desc1 = patches(gray1, keypoints1, patchSize=patchSize)
        kp2desc, desc2 = patches(gray2, keypoints2, patchSize=patchSize)

    if len(desc1) == 0 or len(desc2) == 0:
        raise RuntimeError("No valid descriptors were extracted.")

    if matchChoice == "correlation":
        matches = bestMatchesCorr(desc1, desc2, numMatches=numMatches)
    elif matchChoice == "euclidean":
        matches = bestMatchesEuc(desc1, desc2, ratio=ratio, numMatches=numMatches)
    else:
        raise ValueError("matchChoice must be 'correlation' or 'euclidean'")

    if len(matches) < max(3, ransac_sample_size):
        raise RuntimeError("Not enough matches for RANSAC.")

    M, inlier_mask, pts1, pts2 = ransac_affine(
        kp1desc,
        kp2desc,
        matches,
        n_iter=ransac_iter,
        threshold=ransac_threshold,
        n_sample=ransac_sample_size,
    )

    if M is None:
        raise RuntimeError("RANSAC failed to estimate an affine transform.")

    n_in = int(inlier_mask.sum())
    n_out = int((~inlier_mask).sum())
    inlier_matches = [m for m, f in zip(matches, inlier_mask) if f]
    residuals = affineError(M, kp1desc, kp2desc, inlier_matches)
    avg_residual = float(np.mean(residuals)) if len(residuals) > 0 else np.inf
    accuracy = compute_accuracy_score(M, kp1desc, kp2desc, inlier_matches)

    panorama = None
    if show_panorama:
        panorama = warpStitch(image1, image2, M)
        plt.figure(figsize=(14, 7))
        plt.imshow(panorama)
        plt.title("Final Panorama")
        plt.axis("off")
        plt.show()

    result = {
        "M": M,
        "matches": matches,
        "inlier_mask": inlier_mask,
        "inlier_matches": inlier_matches,
        "n_matches": len(matches),
        "n_inliers": n_in,
        "n_outliers": n_out,
        "avg_residual": avg_residual,
        "accuracy": accuracy,
        "panorama": panorama,
        "kp1desc": kp1desc,
        "kp2desc": kp2desc,
        "harrisResponse1": harrisResponse1,
        "harrisResponse2": harrisResponse2,
        "config": {
            "min_distance": min_distance,
            "num_peaks": num_peaks,
            "threshold_rel": threshold_rel,
            "useSift": useSift,
            "patchSize": patchSize,
            "matchChoice": matchChoice,
            "ratio": ratio,
            "numMatches": numMatches,
            "ransac_iter": ransac_iter,
            "ransac_threshold": ransac_threshold,
            "ransac_sample_size": ransac_sample_size,
        }
    }

    if print_summary:
        print(f"Matches: {result['n_matches']}")
        print(f"Inliers: {result['n_inliers']}  Outliers: {result['n_outliers']}")
        print(f"Average inlier residual: {result['avg_residual']:.3f} px")
        print(f"Accuracy score: {result['accuracy']:.3f} px")
        print(f"Estimated transform:\n{result['M']}")

    return result


def run_sensitivity_analysis(image1_path, image2_path, parameter_name, values, base_config=None, plot=True):
    
    if base_config is None:
        base_config = {}

    image1, image2, gray1, gray2 = load_images(image1_path, image2_path)
    results = []

    for value in values:
        current_config = dict(base_config)
        current_config[parameter_name] = value

        try:
            run = pipeline(
                image1=image1,
                image2=image2,
                gray1=gray1,
                gray2=gray2,
                show_panorama=False,
                print_summary=False,
                **current_config,
            )
            results.append({
                "value": value,
                "accuracy": run["accuracy"],
                "n_inliers": run["n_inliers"],
                "n_outliers": run["n_outliers"],
                "avg_residual": run["avg_residual"],
                "n_matches": run["n_matches"],
            })
            print(
                f"{parameter_name}={value} -> accuracy={run['accuracy']:.3f}, "
                f"inliers={run['n_inliers']}, matches={run['n_matches']}"
            )
        except Exception as e:
            results.append({
                "value": value,
                "accuracy": np.inf,
                "n_inliers": 0,
                "n_outliers": 0,
                "avg_residual": np.inf,
                "n_matches": 0,
                "error": str(e),
            })
            print(f"{parameter_name}={value} -> FAILED ({e})")

    if plot:
        plot_sensitivity_results(results, parameter_name)

    return results


def plot_sensitivity_results(results, parameter_name):
    valid_results = [r for r in results if np.isfinite(r["accuracy"])]
    if len(valid_results) == 0:
        print("No valid runs to plot.")
        return

    x = [r["value"] for r in valid_results]
    y = [r["accuracy"] for r in valid_results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy score (mean Euclidean distance in px)")
    plt.title(f"Sensitivity analysis for {parameter_name}")
    plt.grid(True)
    plt.show()


def run_assignment_sensitivity_suite(image1_path, image2_path):
    base = {
        "useSift": False,
        "patchSize": 11,
        "matchChoice": "euclidean",
        "ratio": 0.75,
        "numMatches": 200,
        "ransac_iter": 1000,
        "ransac_threshold": 3.0,
        "ransac_sample_size": 4,
        "min_distance": 4,
        "num_peaks": 500,
        "threshold_rel": 0.001,
    }

    analyses = {
        "patchSize": [7, 11, 15, 21],
        "numMatches": [50, 100, 150, 200, 300],
        "ratio": [0.60, 0.70, 0.75, 0.80, 0.90],
        "ransac_threshold": [1.5, 2.0, 3.0, 4.0, 5.0],
        "ransac_iter": [250, 500, 1000, 2000],
    }

    all_results = {}
    for parameter_name, values in analyses.items():
        print("\n" + "=" * 60)
        print(f"Running sensitivity analysis for {parameter_name}")
        all_results[parameter_name] = run_sensitivity_analysis(
            image1_path=image1_path,
            image2_path=image2_path,
            parameter_name=parameter_name,
            values=values,
            base_config=base,
            plot=True,
        )

    return all_results


if __name__ == "__main__":
    image1Location = 'one.jpg'
    image2Location = 'two.jpg'

    image1, image2, gray1, gray2 = load_images(image1Location, image2Location)

    # Single run
    result = pipeline(
        image1=image1,
        image2=image2,
        gray1=gray1,
        gray2=gray2,
        useSift=False,
        patchSize=11,
        matchChoice="euclidean",
        ratio=0.75,
        numMatches=200,
        ransac_iter=1000,
        ransac_threshold=3.0,
        show_panorama=True,
        print_summary=True,
    )

    suite_results = run_assignment_sensitivity_suite(image1Location, image2Location)
