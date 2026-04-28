"""
IoT Dual-Camera Multi-View Image Quality Enhancement  — v4 (Stability Fix)
==========================================================================

Fixes:
  1. KeyError: 'weight_vis' resolved by restoring the weight map in run_pipeline.
  2. "Spiral Noise" artifact prevention: 
     - Reverted from Homography (3x3) to limited Affine (2x3, rotation/scale/translation only).
     - Added a "Sanity Check" on the transformation matrix to detect and reject extreme warps.
     - Enhanced SIFT + FLANN matching for better accuracy than ORB.

Pipeline:
  S1 – EXIF-correct load
  S2 – Stable Global Alignment (SIFT + RANSAC Affine) 
  S3 – Sharpness-informed Weight Map (Laplacian Variance)
  S4 – Frequency Domain Fusion (Detail Injection)
  S5 – Edge-aware Sharpening & CLAHE Post-processing
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import io

st.set_page_config(page_title="IoT Camera Enhancer Pro", layout="wide")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_image(file) -> np.ndarray:
    """Load image with correct EXIF orientation."""
    pil = Image.open(file)
    pil = ImageOps.exif_transpose(pil)
    rgb = np.array(pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(to_rgb(bgr))

# ─────────────────────────────────────────────────────────────
# Stage 2 — Stable Global Alignment
# ─────────────────────────────────────────────────────────────

def stable_alignment(m1: np.ndarray, m2: np.ndarray, max_features: int = 4000):
    """
    Computes a stable 2x3 Affine matrix to align m2 to m1.
    Uses SIFT for higher quality matches than ORB.
    """
    g1 = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create(nfeatures=max_features)
    kp1, d1 = detector.detectAndCompute(g1, None)
    kp2, d2 = detector.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0

    # FLANN Matcher for SIFT
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d2, d1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 10:
        return None, len(good)

    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # estimateAffinePartial2D is MUCH more stable for handheld shots.
    # It only allows Rotation, Translation, and Uniform Scaling (4 DOF).
    # This prevents the "spiral/vortex" artifacts caused by Homography's 8 DOF.
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if M is not None:
        # Sanity check: ensure scale is reasonable (0.1x to 10x)
        scale = np.sqrt(M[0,0]**2 + M[0,1]**2)
        if scale < 0.1 or scale > 10.0:
            return None, 0
            
    n_in = int(mask.sum()) if mask is not None else 0
    return M, n_in

def warp_aligned(img: np.ndarray, M: np.ndarray, target_shape):
    h, w = target_shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

# ─────────────────────────────────────────────────────────────
# Stage 3 — Sharpness & Fusion
# ─────────────────────────────────────────────────────────────

def get_sharpness_map(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # Variance in local neighborhood
    mu = cv2.GaussianBlur(lap, (21, 21), 0)
    mu2 = cv2.GaussianBlur(lap**2, (21, 21), 0)
    sigma = cv2.sqrt(cv2.max(0, mu2 - mu**2))
    return cv2.normalize(sigma, None, 0, 1, cv2.NORM_MINMAX)

def lab_frequency_fusion(roi_blurry, roi_sharp, weight_map):
    """Blends sharp details into blurry base with proper L-channel clipping."""
    # Convert to LAB for luminance-based fusion
    lab_b = cv2.cvtColor(roi_blurry, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.cvtColor(roi_sharp, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    l_b, a_b, b_b = cv2.split(lab_b)
    l_s, _, _ = cv2.split(lab_s)
    
    # Detail = (SharpL - BlurredVersionOfSharpL)
    l_s_smooth = cv2.GaussianBlur(l_s, (15, 15), 0)
    detail = l_s - l_s_smooth
    
    # Use weight map to decide where to inject
    w = cv2.GaussianBlur(weight_map, (31, 31), 0)
    
    # Inject details into blurry L
    # FIX: Use 0-255 for L channel clipping in OpenCV LAB uint8 domain
    enhanced_l = l_b + (detail * w * 1.0)
    enhanced_l = np.clip(enhanced_l, 0, 255)
    
    res_lab = cv2.merge([enhanced_l, a_b, b_b])
    return cv2.cvtColor(res_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def refine_with_flow(img1, warped_img2):
    """Refine alignment using dense Optical Flow for per-pixel mapping."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    
    # Farneback Optical Flow
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = g1.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x - flow[..., 0]
    map_y = y - flow[..., 1]
    
    return cv2.remap(warped_img2, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

# ─────────────────────────────────────────────────────────────
# Full run_pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline(m1, m2, roi, max_features, clahe_clip, sharp_str, use_flow=True):
    h1, w1 = m1.shape[:2]
    
    # S2: Global Alignment
    M, n_in = stable_alignment(m1, m2, max_features)
    
    if M is not None:
        warped_m2 = warp_aligned(m2, M, m1.shape)
        aligned = True
        # S2.1: Per-Pixel Refinement (Optical Flow)
        if use_flow:
            warped_m2 = refine_with_flow(m1, warped_m2)
    else:
        warped_m2 = cv2.resize(m2, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        aligned = False

    # S3: ROI Extraction
    x1, y1, x2, y2 = roi
    roi_m1 = m1[y1:y2, x1:x2]
    roi_m2 = warped_m2[y1:y2, x1:x2]
    
    # Generate Weight Map based on relative sharpness
    s1 = get_sharpness_map(roi_m1)
    s2 = get_sharpness_map(roi_m2)
    
    # Weight map: high where m2 is sharper than m1
    weight = (s2 - s1)
    weight = np.clip(weight, 0, 1)
    
    # S4: Fusion
    enhanced_roi = lab_frequency_fusion(roi_m1, roi_m2, weight)
    
    # S5: Post-processing
    # CLAHE
    lab = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_roi = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Final Sharpen
    blur = cv2.GaussianBlur(enhanced_roi, (0, 0), 1.5)
    enhanced_roi = cv2.addWeighted(enhanced_roi, 1.0 + sharp_str, blur, -sharp_str, 0)

    # Result - Blend with soft mask to avoid "black rectangle" or hard edges
    result = m1.copy()
    
    # Create soft mask for ROI blending
    mask = np.zeros((h1, w1), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    
    # Expand enhanced_roi back to full size temporarily for blending OR do it locally
    # It's easier to do it locally if we feather the transition
    
    # For now, let's keep it simple but fix the shadow by ensuring we don't have hard boundaries
    # A more robust way is to use the full image for mask blending
    full_enhanced = m1.copy()
    full_enhanced[y1:y2, x1:x2] = enhanced_roi
    
    # Blend: result = (1-mask)*original + mask*enhanced
    mask_3c = cv2.merge([mask, mask, mask])
    result = (m1.astype(np.float32) * (1.0 - mask_3c) + full_enhanced.astype(np.float32) * mask_3c).astype(np.uint8)
    
    # Weight Visualization
    weight_vis = cv2.applyColorMap((weight * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return dict(
        result=result,
        warped_m2_full=warped_m2,
        warped_roi=roi_m2,
        weight_vis=weight_vis,
        n_inliers=n_in,
        aligned=aligned
    )

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.title("🛡️ Stable IoT Camera Image Quality Enhancer")
st.caption("Algorithm: SIFT Affine Alignment + L-channel Detail Injection + LAB Contrast Fix")

with st.sidebar:
    st.header("ROI Selection (%)")
    rx1 = st.slider("Left %", 0, 90, 2)
    rx2 = st.slider("Right %", 10, 100, 70)
    ry1 = st.slider("Top %", 0, 80, 0)
    ry2 = st.slider("Bottom %", 10, 100, 45)

    st.header("Parameters")
    max_f = st.select_slider("SIFT Features", options=[1000, 2000, 4000, 8000], value=4000)
    clahe = st.slider("Contrast (CLAHE)", 1.0, 5.0, 2.5)
    sharp = st.slider("Edge Strength", 0.0, 3.0, 1.2)
    
    st.header("Refinement")
    use_flow = st.checkbox("Enable Per-Pixel Refinement (Flow)", value=True)
    show_w = st.checkbox("Show Sharpness Weight Map", value=True)

c1, c2 = st.columns(2)
with c1: f1 = st.file_uploader("Image 1 (M1 - Blurry)", type=["jpg","png","jpeg"])
with c2: f2 = st.file_uploader("Image 2 (M2 - Sharp Ref)", type=["jpg","png","jpeg"])

if f1 and f2:
    m1 = load_image(f1)
    m2 = load_image(f2)
    h, w = m1.shape[:2]
    
    roi_box = (int(rx1/100*w), int(ry1/100*h), int(rx2/100*w), int(ry2/100*h))
    
    with st.spinner("Processing Stable Alignment & Fusion..."):
        out = run_pipeline(m1, m2, roi_box, max_f, clahe, sharp, use_flow=use_flow)
        
    res = out["result"]
    
    if out["aligned"]:
        st.success(f"Alignment Success: {out['n_inliers']} features matched.")
    else:
        st.warning("Alignment Failed: Using fallback resize. Results may be misaligned.")

    # Results display
    st.subheader("Enhanced Output")
    st.image(to_rgb(res), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Original ROI (M1)")
        st.image(to_rgb(m1[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]), use_container_width=True)
    with col2:
        st.write("Enhanced ROI")
        st.image(to_rgb(res[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]), use_container_width=True)
    with col3:
        st.write("Aligned Ref ROI (M2)")
        st.image(to_rgb(out["warped_roi"]), use_container_width=True)

    if show_w:
        st.write("Sharpness Map (Blue=Keep M1, Red=Take M2)")
        st.image(to_rgb(out["weight_vis"]), use_container_width=True)

    # Download
    buf = io.BytesIO()
    to_pil(res).save(buf, format="JPEG", quality=95)
    st.download_button("Download Enhanced Image", buf.getvalue(), "enhanced_m1.jpg", "image/jpeg")
else:
    st.info("Upload M1 and M2 images to begin.")