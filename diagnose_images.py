import cv2
import numpy as np
import os

def analyze_file(filepath, label):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[{label}] Failed to load.")
        return
    
    print(f"\n--- {label} Analysis ({os.path.basename(filepath)}) ---")
    print(f"Shape: {img.shape}, Dtype: {img.dtype}")
    
    # 1. Active Bits (Bit Cliff)
    active_bits = 0
    for b in range(16):
        if np.any(img & (1 << b)):
            active_bits = b + 1
    print(f"Max Active Bit: {active_bits}")
    
    # 2. Unique Values in a gradient patch
    # Let's take a patch of the sky (center-ish)
    y, x = img.shape[0]//4, img.shape[1]//2
    patch = img[y:y+100, x:x+100]
    unique_vals = len(np.unique(patch))
    print(f"Unique values in 100x100 sky patch: {unique_vals}")
    
    # 3. Gap Analysis
    # Let's look at the difference between sorted unique values in the patch
    flat_patch = np.unique(patch.flatten())
    if len(flat_patch) > 1:
        gaps = np.diff(flat_patch)
        unique_gaps, gap_counts = np.unique(gaps, return_counts=True)
        print(f"Common gaps between unique color values: {dict(zip(unique_gaps[:5], gap_counts[:5]))}")
    
    # 4. LSB Noise Floor
    # Is the lowest bit just random noise or is it correlated?
    lsb = img & 1
    lsb_mean = np.mean(lsb)
    print(f"LSB Mean Activity: {lsb_mean:.4f}")

file_a = "/Users/doug/.gemini/antigravity/brain/7e0980d3-18b1-499d-a8c1-d11a6cd64350/uploaded_image_0_1767660112603.png"
file_b = "/Users/doug/.gemini/antigravity/brain/7e0980d3-18b1-499d-a8c1-d11a6cd64350/uploaded_image_1_1767660112603.png"

analyze_file(file_a, "FILE A (8-bit upscaled)")
analyze_file(file_b, "FILE B (Native 16-bit)")
