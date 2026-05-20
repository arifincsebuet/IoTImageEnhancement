import cv2
import numpy as np
import os

def stable_affine_alignment(m1, m2, max_features=5000):
    g1 = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0
    bf = cv2.BFMatcher()
    pairs = bf.knnMatch(d2, d1, k=2)
    good = []
    for m, n in pairs:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) < 10:
        return None, len(good)
    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    n_in = int(mask.sum()) if mask is not None else 0
    return M, n_in

def refine_with_flow(img1, warped_img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = g1.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x - flow[..., 0]
    map_y = y - flow[..., 1]
    return cv2.remap(warped_img2, map_x, map_y, cv2.INTER_LANCZOS4)

def laplacian_variance_map(gray, ksize=15):
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    sq = (lap ** 2).astype(np.float32)
    return cv2.GaussianBlur(sq, (ksize, ksize), 0)

def lab_frequency_fusion(roi_blurry, roi_sharp, weight_map):
    """Blends sharp details into blurry base with proper L-channel clipping."""
    lab_b = cv2.cvtColor(roi_blurry, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.cvtColor(roi_sharp, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_b, A_b, B_b = cv2.split(lab_b)
    L_s, _, _ = cv2.split(lab_s)
    
    l_s_smooth = cv2.GaussianBlur(L_s, (15, 15), 0)
    details = L_s - l_s_smooth
    
    w = cv2.GaussianBlur(weight_map, (31, 31), 0)
    enhanced_L = L_b + (details * w * 1.0)
    enhanced_L = np.clip(enhanced_L, 0, 255)
    
    result = cv2.merge([enhanced_L, A_b, B_b])
    return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)

def main():
    m1 = cv2.imread("m1.jpg")
    m2 = cv2.imread("m2.jpg")
    if m1 is None or m2 is None:
        print("Images not found.")
        return

    print("Step 1: Stable Affine Alignment...")
    M, n_in = stable_affine_alignment(m1, m2)
    if M is None:
        print("Alignment failed.")
        return
    
    warped_base = cv2.warpAffine(m2, M, (m1.shape[1], m1.shape[0]), flags=cv2.INTER_LANCZOS4)
    
    print("Step 2: Optical Flow Refinement...")
    refined_full = refine_with_flow(m1, warped_base)
    
    # Use the same calendar ROI as mentioned in app.py logic/user request
    # Approx. Left 2%, Right 70%, Top 0%, Bottom 45%
    h, w = m1.shape[:2]
    x1, y1, x2, y2 = int(0.02*w), int(0.0*h), int(0.70*w), int(0.45*h)
    
    roi_m1 = m1[y1:y2, x1:x2]
    roi_ref = refined_full[y1:y2, x1:x2]
    
    # Local weight map: high where ref is sharper than base in absolute terms
    g_m1 = cv2.cvtColor(roi_m1, cv2.COLOR_BGR2GRAY)
    g_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
    s1 = laplacian_variance_map(g_m1)
    s2 = laplacian_variance_map(g_ref)
    diff = np.clip(s2 - s1, 0, None)
    max_val = diff.max()
    if max_val > 1e-5:
        weight = diff / max_val
    else:
        weight = np.zeros_like(diff)
    
    print("Step 3: LAB Frequency Fusion...")
    fused_roi = lab_frequency_fusion(roi_m1, roi_ref, weight)
    
    print("Step 4: Post-processing...")
    lab = cv2.cvtColor(fused_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    enhanced_roi = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
    
    # Sharpen
    blur = cv2.GaussianBlur(enhanced_roi, (0, 0), 1.5)
    final_roi = cv2.addWeighted(enhanced_roi, 2.2, blur, -1.2, 0)
    
    # Result blending with soft mask
    result = m1.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    
    full_enhanced = m1.copy()
    full_enhanced[y1:y2, x1:x2] = final_roi
    
    mask_3c = cv2.merge([mask, mask, mask])
    result = (m1.astype(np.float32) * (1.0 - mask_3c) + full_enhanced.astype(np.float32) * mask_3c).astype(np.uint8)
    
    cv2.imwrite("enhanced_result_v4.jpg", result)
    print("Success! Result saved to enhanced_result_v4.jpg")

if __name__ == "__main__":
    main()
