import numpy as np
import cv2
from scipy.fft import fft

def load_image(filepath: str) -> np.ndarray:
    """
    Loads an image in strict mode (UNCHANGED).
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")
    return img

def analyze_bit_depth(img: np.ndarray) -> dict:
    """
    Analyzes which bits are active. Supports 8 and 16 bit containers.
    """
    is_16bit = (img.dtype == np.uint16)
    container_depth = 16 if is_16bit else 8
    
    flat_pixels = img.flatten()
    accumulated_bits = np.bitwise_or.reduce(flat_pixels)
    
    active_bits = []
    for i in range(container_depth):
        if (accumulated_bits >> i) & 1:
            active_bits.append(i)
            
    if not active_bits:
        return {"effective_depth": 0, "active_bits": [], "padding": "N/A"}

    max_active = max(active_bits)
    min_active = min(active_bits)
    
    return {
        "container_depth": container_depth,
        "effective_depth": max_active + 1,
        "lowest_active_bit": min_active,
        "is_padded": min_active > 0
    }

def detect_histogram_combing(img: np.ndarray) -> dict:
    """
    Bit-Depth Aware Histogram analysis.
    Checks for the "Combing" effect (upscaling gaps).
    """
    is_16bit = (img.dtype == np.uint16)
    bins = 65536 if is_16bit else 256
    
    # Handle Grayscale conversion (including RGBA support)
    if len(img.shape) >= 3:
        n_channels = img.shape[2]
        if n_channels == 3:
            lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif n_channels == 4:
            lum = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            lum = img[:,:,0] # Fallback
    else:
        lum = img

    hist = cv2.calcHist([lum], [0], None, [bins], [0, bins])
    hist = hist.flatten()
    
    non_zero_bins = np.count_nonzero(hist)
    unique_ratio = non_zero_bins / float(bins)
    
    # FFT Check is only truly reliable for 16-bit upscaling (8->16)
    # For 8-bit, we just look at the unique ratio as a proxy for quantization
    comb_strength = 0.0
    is_upscaled = False
    
    if is_16bit:
        # Normalize histogram
        hist_norm = hist - np.mean(hist)
        fft_vals = np.abs(fft(hist_norm))
        fft_vals = fft_vals[:len(fft_vals)//2]
        
        target_freq_index = 255 # Spike at ~256 gaps
        window = 10
        local_region = fft_vals[target_freq_index-window : target_freq_index+window]
        peak_val = np.max(local_region)
        avg_val = np.mean(fft_vals)
        comb_strength = float(peak_val / (avg_val + 1e-6))
        is_upscaled = comb_strength > 50.0
    else:
        # For 8-bit, if unique ratio is very low (e.g. < 0.5), it's heavily posterized
        is_upscaled = unique_ratio < 0.5

    return {
        "unique_values_ratio": float(round(unique_ratio, 4)),
        "comb_strength_score": float(round(comb_strength, 2)),
        "likely_upscaled_8bit": bool(is_upscaled)
    }
