import numpy as np
import cv2

def generate_sigmoid_lut(intensity: float = 10.0, bit_depth: int = 16) -> np.ndarray:
    """
    Generates a Sigmoid LUT. We use floating point for the math to ensure precision.
    """
    max_val = (2**bit_depth) - 1
    x = np.linspace(0, 1, max_val + 1)
    
    # Sigmoid: k controls the contrast boost
    k = intensity
    sigmoid = 1 / (1 + np.exp(-k * (x - 0.5)))
    
    # Normalize 0.0 to 1.0
    s_min, s_max = sigmoid.min(), sigmoid.max()
    sigmoid_norm = (sigmoid - s_min) / (s_max - s_min)
    
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    return (sigmoid_norm * max_val).astype(dtype)

def compute_texture_mask(img: np.ndarray) -> np.ndarray:
    """
    Identifies high-frequency texture (edges, rocks, noise) on a BASE image.
    This mask is used to exclude busy regions from banding analysis.
    """
    is_16bit = (img.dtype == np.uint16)
    
    if len(img.shape) >= 3:
        n_channels = img.shape[2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if n_channels == 3 else cv2.COLOR_BGRA2GRAY)
    else:
        gray = img
        
    img_f = gray.astype(np.float32)
    laplacian = cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)
    abs_laplacian = np.abs(laplacian)
    
    # Baseline threshold for native image texture
    # 16-bit: 500 is a safe bet for "significant" detail
    # 8-bit: 5 is roughly equivalent
    thresh = 500 if is_16bit else 5
    mask = abs_laplacian > thresh
    
    # Dilate to ensure we cover the vicinity of textures
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

def detect_banding_artifacts(img: np.ndarray, external_texture_mask: np.ndarray = None) -> dict:
    """
    Heuristic 6.1: Stable Texture-Aware Banding Detection.
    Uses an external texture mask (from the original image) to ensure 
    that increasing contrast doesn't 'hide' banding by turning it into 'texture'.
    """
    is_16bit = (img.dtype == np.uint16)
    
    if len(img.shape) >= 3:
        n_channels = img.shape[2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if n_channels == 3 else cv2.COLOR_BGRA2GRAY)
    else:
        gray = img

    img_f = gray.astype(np.float32)
    h, w = gray.shape
    block_size = 16
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    banding_mask = np.zeros_like(gray, dtype=bool)
    
    # Global Jumps
    # Calc diffs on the STRESSED image
    diff_h = np.abs(img_f[:, 1:] - img_f[:, :-1])
    diff_v = np.abs(img_f[1:, :] - img_f[:-1, :])
    
    banded_count = 0
    candidate_tiles = 0
    
    for r_idx in range(h_blocks):
        for c_idx in range(w_blocks):
            r_start = r_idx * block_size
            c_start = c_idx * block_size
            
            # Check Texture Mask (if provided)
            if external_texture_mask is not None:
                t_mask = external_texture_mask[r_start:r_start+block_size, c_start:c_start+block_size]
                if np.mean(t_mask) > 0.3: # Skip if 30% of tile is texture
                    continue
            
            tile = gray[r_start : r_start+block_size, c_start : c_start+block_size]
            t_range = np.ptp(tile)
            
            # Sensitivity Floor: If range is ultra low, it's just a flat solid
            # 128 in 16-bit is effectively nothing
            if t_range < (128 if is_16bit else 1):
                continue
            
            candidate_tiles += 1
            
            # Extract Local Jumps
            t_diff_h = diff_h[r_start : r_start+block_size, c_start : c_start+block_size-1]
            t_diff_v = diff_v[r_start : r_start+block_size-1, c_start : c_start+block_size]
            
            # 8-bit Step Jump (256/257)
            jump_limit = 256 if is_16bit else 2
            
            sig_jumps = np.sum(t_diff_h >= jump_limit) + np.sum(t_diff_v >= jump_limit)
            total_changes = np.sum(t_diff_h > 0) + np.sum(t_diff_v > 0)
            
            activity = total_changes / float(tile.size * 2)
            severity = sig_jumps / (total_changes + 1e-6)
            
            # Banding Logic: Low activity (plateaus) + High jump severity (8-bit steps)
            if is_16bit:
                is_banded = (activity < 0.25) and (severity > 0.6)
            else:
                is_banded = (activity < 0.05) or (severity > 0.9)
                
            if is_banded:
                banding_mask[r_start:r_start+block_size, c_start:c_start+block_size] = True
                banded_count += 1
                
    score = (banded_count / (candidate_tiles + 1e-6)) * 100.0
    
    return {
        "banding_score": float(round(score, 4)),
        "is_severe": bool(score > 5.0),
        "heatmap_mask": banding_mask
    }

def run_stress_test(img: np.ndarray, intensity: float = 15.0) -> dict:
    is_16bit = (img.dtype == np.uint16)
    bit_depth = 16 if is_16bit else 8
    
    # 1. Compute Texture Mask on the ORIGINAL image (Stable forensic context)
    tex_mask = compute_texture_mask(img)
    
    # 2. Apply S-Curve Contrast Stress
    lut = generate_sigmoid_lut(intensity=intensity, bit_depth=bit_depth)
    
    if len(img.shape) == 3:
        ch = img.shape[2]
        res_img = np.zeros_like(img)
        for i in range(ch):
            if ch == 4 and i == 3: # Skip transparency
                res_img[:,:,i] = img[:,:,i]
            else:
                res_img[:,:,i] = lut[img[:,:,i]]
    else:
        res_img = lut[img]
        
    # 3. Detect Banding using the stable mask
    analysis = detect_banding_artifacts(res_img, external_texture_mask=tex_mask)
    
    return {
        "stress_intensity": intensity,
        "banding_metric": analysis["banding_score"],
        "passed": not analysis["is_severe"],
        "heatmap_mask": analysis["heatmap_mask"],
        "stressed_image_preview": res_img
    }
