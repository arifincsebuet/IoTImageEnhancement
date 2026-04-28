# 🛡️ IoT Dual-Camera Image Quality Enhancement

A research project for improving image quality by fusing two views from multiple IoT cameras. The pipeline uses geometric alignment, frequency-domain detail injection, and contrast enhancement to produce a sharper, higher-quality output image from a blurry/sharp camera pair.

---

## 📌 Overview

Modern IoT deployments often use low-cost cameras that produce blurry or noisy images under challenging conditions. This project leverages a **second reference camera** (capturing the same scene from a slightly different angle) to inject missing high-frequency details back into the primary (blurry) image.

The approach is entirely **matrix-based**, relying on geometric transformations (affine alignment), Laplacian-variance sharpness maps, and LAB-space frequency fusion — making it well-suited for real-time or near-real-time IoT pipelines.

---

## ✨ Key Features

- **EXIF-Aware Loading** — Automatically corrects image rotation from camera metadata.
- **Stable Global Alignment (SIFT + RANSAC Affine)** — Uses Scale-Invariant Feature Transform (SIFT) with FLANN matching and a partial 2D affine estimator (4 DOF: rotation, translation, uniform scale) to avoid spiral/vortex artifacts caused by unconstrained homography.
- **Sanity-Checked Transformation** — Rejects any estimated transform whose scale factor falls outside a safe range (0.1× – 10×).
- **Per-Pixel Optical Flow Refinement** — Farneback dense optical flow corrects residual sub-pixel misalignment after global alignment.
- **Sharpness-Informed Weight Map** — A Laplacian-variance map identifies which pixels in the reference image are sharper than the corresponding pixels in the base image.
- **LAB Frequency-Domain Fusion** — Detail injection operates in the luminance (L) channel of the LAB color space, preserving original color information while sharpening edges.
- **CLAHE Contrast Enhancement** — Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied post-fusion to boost local contrast without clipping.
- **Edge-Aware Unsharp Masking** — A final sharpening pass (Gaussian-subtraction weighted blend) enhances perceived sharpness.
- **Soft ROI Blending** — A feathered mask prevents visible hard edges between the enhanced region of interest (ROI) and the unmodified image background.

---

## 🔬 Pipeline Architecture

```
Input: M1 (blurry base)  +  M2 (sharp reference)
        │                        │
        └──────────┬─────────────┘
                   │
         S1 │ EXIF-Correct Load
                   │
         S2 │ Stable Global Alignment
            │   SIFT keypoints → FLANN → Lowe's ratio test
            │   estimateAffinePartial2D (RANSAC)
            │   Sanity check on scale factor
                   │
       S2.1 │ Optical Flow Refinement (Farneback)
                   │
         S3 │ ROI Extraction + Sharpness Weight Map
            │   Laplacian Variance → per-pixel weight
                   │
         S4 │ LAB Frequency Fusion
            │   Detail = L_sharp − GaussBlur(L_sharp)
            │   Enhanced_L = L_blurry + (Detail × Weight)
                   │
         S5 │ Post-processing
            │   CLAHE (L channel) → Unsharp Mask
            │   Soft-mask blending into full image
                   │
Output: Enhanced M1 (full resolution, JPEG)
```

---

## 📁 Project Structure

```
ImageQualityImprovement/
│
├── app.py               # Interactive Streamlit web application
├── verify_logic.py      # Standalone command-line pipeline script
├── compute_metrics.py   # Sharpness benchmarking utility
│
├── m1.jpg               # Sample input: blurry base image
├── m2.jpg               # Sample input: sharp reference image
│
└── README.md            # This file
```

---

## 🗂️ Module Descriptions

### `app.py` — Streamlit Web Interface

An interactive browser-based application for exploring the full enhancement pipeline.

**UI Controls (Sidebar):**
| Parameter | Description |
|---|---|
| **ROI Selection (%)** | Define the region of interest (left, right, top, bottom) as percentages of image dimensions |
| **SIFT Features** | Number of SIFT keypoints to detect (1,000 – 8,000). More features → better alignment, slower speed |
| **Contrast (CLAHE)** | Clip limit for adaptive histogram equalization (1.0 – 5.0) |
| **Edge Strength** | Unsharp masking weight (0.0 – 3.0) |
| **Per-Pixel Refinement (Flow)** | Toggle Farneback optical flow refinement on/off |
| **Show Sharpness Weight Map** | Visualize the pixel-level sharpness decision map (Blue = keep M1, Red = take from M2) |

**Outputs displayed:**
- Full-resolution enhanced image
- Side-by-side ROI comparison (original M1, enhanced, aligned M2 reference)
- Sharpness weight map (JET colormap)
- Downloadable enhanced JPEG

---

### `verify_logic.py` — Command-Line Pipeline

A headless script that runs the full pipeline end-to-end without a UI. Reads `m1.jpg` and `m2.jpg` from the current directory and saves the result as `enhanced_result_v4.jpg`.

**Steps executed:**
1. Stable affine alignment (SIFT + BFMatcher + RANSAC)
2. Optical flow refinement
3. Laplacian-variance weight map computation
4. LAB frequency fusion
5. CLAHE + unsharp masking
6. Soft-mask blending

**Usage:**
```bash
python verify_logic.py
```

Output: `enhanced_result_v4.jpg`

---

### `compute_metrics.py` — Sharpness Benchmark

Quantifies the improvement in sharpness introduced by the pipeline by computing the **Laplacian variance** of the ROI in the original and enhanced images. A higher Laplacian variance indicates a sharper image.

**Usage:**
```bash
python compute_metrics.py
```

**Prerequisites:** `m1.jpg` and `enhanced_result_v4.jpg` must exist in the working directory (run `verify_logic.py` first).

**Sample Output:**
```
=== ROI Sharpness Benchmark ===
Original Blurry Base (M1)   : 42.87
Enhanced Output (Result)    : 118.64
Sharpness Improvement       : +176.73%
```

---

## ⚙️ Requirements

| Package | Purpose |
|---|---|
| `opencv-python` | Image processing, SIFT, alignment, optical flow, CLAHE |
| `numpy` | Matrix operations and array math |
| `Pillow` | EXIF-aware image loading |
| `streamlit` | Interactive web application (`app.py` only) |

Install all dependencies:
```bash
pip install opencv-python numpy Pillow streamlit
```

> **Note:** `cv2.SIFT_create()` requires `opencv-contrib-python` in older OpenCV versions. If you encounter an `AttributeError`, install it with:
> ```bash
> pip install opencv-contrib-python
> ```

---

## 🚀 Getting Started

### Option A: Interactive Web App
```bash
streamlit run app.py
```
Open the URL shown in the terminal (typically `http://localhost:8501`) and upload your M1 and M2 images through the interface.

### Option B: Command-Line Script
1. Place your images as `m1.jpg` (blurry base) and `m2.jpg` (sharp reference) in the project directory.
2. Run the pipeline:
   ```bash
   python verify_logic.py
   ```
3. Evaluate sharpness improvement:
   ```bash
   python compute_metrics.py
   ```

---

## 🧮 Core Equations

**Laplacian Variance (Sharpness Proxy):**
$$\sigma^2_{\text{lap}} = \text{Var}\!\left(\nabla^2 I\right)$$

**Weight Map (per-pixel sharpness advantage of M2 over M1):**
$$W(x,y) = \text{clip}\!\left(\sigma^2_{M2}(x,y) - \sigma^2_{M1}(x,y),\ 0,\ 1\right)$$

**LAB Detail Injection:**
$$L_{\text{enhanced}} = L_{M1} + \left(L_{M2} - \widetilde{L}_{M2}\right) \cdot W$$

where $\widetilde{L}_{M2}$ is a Gaussian-blurred version of $L_{M2}$, so $L_{M2} - \widetilde{L}_{M2}$ captures only high-frequency detail.

**Unsharp Masking:**
$$I_{\text{sharp}} = (1 + \alpha)\,I - \alpha\,G_\sigma(I)$$

---

## 📊 Evaluation

The sharpness improvement is measured using the **Laplacian variance** metric over the selected ROI, comparing the original blurry base image against the enhanced output. Typical results show significant gains (>100% sharpness improvement) when M2 is meaningfully sharper than M1 in the ROI.

---

## 🔭 Research Context

This project is part of ongoing research into multi-camera IoT image quality enhancement. The techniques implemented here form the algorithmic foundation described in the associated conference paper, which covers:

- The architectural motivation for a dual-camera IoT setup
- Mathematical formulation of the alignment and fusion stages
- Empirical evaluation of sharpness metrics

---

## 📄 License

This project is for academic research purposes. Please cite this work if you use or adapt this code in your own research.

---

## 👤 Author

Research project — Image Quality Improvement via Multi-View IoT Camera Fusion.
