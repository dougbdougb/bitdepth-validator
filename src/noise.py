import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.stats import pearsonr

def extract_noise_plane(img: np.ndarray) -> np.ndarray:
    """
    Isolates the noise component by subtracting a smoothed version 
    of the image from the original.
    """
    # Use a small Median filter to preserve edges but remove noise
    # We use float32 to allow negative values in the residual
    img_f = img.astype(np.float32)
    
    # 3x3 median blur is effective for fine noise extraction
    # Note: simple blur works best on flat regions, but we use median
    # to avoid capturing high-contrast edges as "noise".
    smoothed = cv2.medianBlur(img, 3).astype(np.float32)
    
    noise_residual = img_f - smoothed
    return noise_residual

def check_noise_periodicity(noise_plane: np.ndarray) -> dict:
    """
    Uses FFT to detect ordered dithering patterns.
    Natural noise is 'white' or 'pink' (random frequencies).
    Dithering shows up as bright stars (spikes) in the FFT.
    """
    # 1. Compute 2D FFT of the noise
    f = fft2(noise_plane)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    
    # 2. Detect Spikes (Peaks)
    # Natural noise is a fuzzy cloud in the center. 
    # Dithering is geometric points outside the center.
    
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Mask out the DC component (center) which is just image brightness
    mask_radius = 5
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 <= mask_radius**2
    magnitude_spectrum[mask] = 0
    
    # Calculate Peak-to-Average Ratio
    avg_energy = np.mean(magnitude_spectrum)
    max_energy = np.max(magnitude_spectrum)
    
    # If the max spike is > 4x the average background energy, it's likely periodic.
    # (Thresholds need tuning on real data, 4.0 is a conservative start)
    spike_ratio = max_energy / (avg_energy + 1e-6)
    
    return {
        "fft_spike_ratio": float(round(spike_ratio, 2)),
        "has_periodic_patterns": bool(spike_ratio > 4.0)
    }

def analyze_cross_channel_correlation(img: np.ndarray) -> dict:
    """
    Checks if noise in R/G/B channels is correlated.
    Real sensors: High correlation (CFA interpolation).
    Fake noise: Often near-zero correlation (random numbers added per channel).
    """
    if len(img.shape) < 3:
        return {"error": "Monochrome image, cannot test channel correlation"}

    # Extract noise planes for each channel
    b_noise = extract_noise_plane(img[:, :, 0]).flatten()
    g_noise = extract_noise_plane(img[:, :, 1]).flatten()
    r_noise = extract_noise_plane(img[:, :, 2]).flatten()
    
    # Calculate Pearson Correlation
    # We use a random subset of pixels to keep this fast
    subset_size = min(50000, len(b_noise))
    indices = np.random.choice(len(b_noise), subset_size, replace=False)
    
    corr_bg, _ = pearsonr(b_noise[indices], g_noise[indices])
    corr_gr, _ = pearsonr(g_noise[indices], r_noise[indices])
    corr_rb, _ = pearsonr(r_noise[indices], b_noise[indices])
    
    avg_corr = (abs(corr_bg) + abs(corr_gr) + abs(corr_rb)) / 3.0
    
    return {
        "avg_channel_correlation": float(round(avg_corr, 4)),
        "interpretation": "Natural" if avg_corr > 0.3 else "Synthetic/Uncorrelated"
    }

def run_noise_analysis(img: np.ndarray) -> dict:
    """
    Orchestrator for noise tests.
    """
    # If RGB, convert to Gray for periodicity check
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        periodicity = check_noise_periodicity(extract_noise_plane(gray))
        correlation = analyze_cross_channel_correlation(img)
    else:
        gray = img
        periodicity = check_noise_periodicity(extract_noise_plane(gray))
        correlation = {"info": "Monochrome - Skipping correlation"}
        
    return {
        **periodicity,
        **correlation
    }
