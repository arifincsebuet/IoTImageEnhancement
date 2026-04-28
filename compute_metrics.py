import cv2
import numpy as np

def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    # Load images
    m1 = cv2.imread("m1.jpg") 
    m2 = cv2.imread("m2.jpg") 
    result = cv2.imread("enhanced_result_v4.jpg") 
    
    if m1 is None or result is None:
        print("Error: Make sure 'm1.jpg' and 'enhanced_result_v4.jpg' exist in this directory.")
        return

    # Define the core ROI evaluated (from verify_logic.py)
    h, w = m1.shape[:2]
    x1, y1, x2, y2 = int(0.02*w), int(0.0*h), int(0.70*w), int(0.45*h)
    
    roi_m1 = m1[y1:y2, x1:x2]
    roi_result = result[y1:y2, x1:x2]
    
    # Sharpness Comparison
    sharp_m1 = compute_sharpness(roi_m1)
    sharp_result = compute_sharpness(roi_result)
    
    print("=== ROI Sharpness Benchmark ===")
    print(f"Original Blurry Base (M1)   : {sharp_m1:.2f}")
    print(f"Enhanced Output (Result)    : {sharp_result:.2f}")
    
    if sharp_m1 > 0:
        improvement = ((sharp_result - sharp_m1) / sharp_m1) * 100
        print(f"Sharpness Improvement       : +{improvement:.2f}%")

if __name__ == "__main__":
    main()
