import numpy as np
import cv2
import os
from src.core import load_image, analyze_bit_depth, detect_histogram_combing
from src.stress import run_stress_test
from src.noise import run_noise_analysis

def generate_synthetic_images():
    """Generates synthetic images for testing different modules."""
    print("Generating synthetic test images...")
    
    # 1. Native 16-bit noise (should pass all checks)
    native = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
    cv2.imwrite("test_native.png", native)
    
    # 2. Upscaled 8-bit (should fail histogram gap check)
    upscaled_8 = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    upscaled = upscaled_8.astype(np.uint16) * 257
    cv2.imwrite("test_upscaled.png", upscaled)
    
    # 3. Banded Gradient (should fail stress test)
    # Create a simple vertical gradient with steps
    gradient_8 = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1)).T
    banded = gradient_8.astype(np.uint16) * 257
    cv2.imwrite("test_banded.png", banded)

    # 4. RGB Periodic Noise (should fail noise check)
    # Adding a simple periodic pattern to noise
    x = np.linspace(0, 10, 512)
    y = np.linspace(0, 10, 512)
    X, Y = np.meshgrid(x, y)
    pattern = (np.sin(X*100) + 1) * 32767
    periodic_noise = (np.random.randint(0, 10000, (512, 512, 3), dtype=np.uint16) + \
                      pattern[:, :, np.newaxis]).astype(np.uint16)
    cv2.imwrite("test_periodic.png", periodic_noise)

    return ["test_native.png", "test_upscaled.png", "test_banded.png", "test_periodic.png"]

def run_suite(filepaths):
    print("\n" + "="*50)
    print("RUNNING ALL ANALYSIS MODULES")
    print("="*50)
    
    for path in filepaths:
        print(f"\n>>> FILE: {path}")
        try:
            img = load_image(path)
            
            # 1. Core
            bd = analyze_bit_depth(img)
            hist = detect_histogram_combing(img)
            print(f"[CORE] Effective Depth: {bd['effective_depth']}-bit | Padded: {bd['is_padded']}")
            print(f"[CORE] Likely Upscaled: {hist['likely_upscaled_8bit']} (Score: {hist['comb_strength_score']})")
            
            # 2. Stress
            stress = run_stress_test(img, intensity=15.0)
            print(f"[STRESS] Banding Metric: {stress['banding_metric']}% | Passed: {stress['passed']}")
            
            # 3. Noise
            noise = run_noise_analysis(img)
            print(f"[NOISE] Periodic Patterns: {noise['has_periodic_patterns']} (Ratio: {noise['fft_spike_ratio']})")
            if 'interpretation' in noise:
                print(f"[NOISE] Interpretation: {noise['interpretation']} (Corr: {noise.get('avg_channel_correlation')})")
                
        except Exception as e:
            print(f"Error analyzing {path}: {e}")

if __name__ == "__main__":
    paths = generate_synthetic_images()
    run_suite(paths)
    
    # Cleanup (optional)
    # for p in paths: os.remove(p)
