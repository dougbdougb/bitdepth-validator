import numpy as np
import cv2
import os
from src.core import load_image, analyze_bit_depth, detect_histogram_combing

def generate_test_image(path, mode='native'):
    """Generates synthetic 16-bit PNG for testing."""
    if mode == 'native':
        # True 16-bit noise
        img = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
    elif mode == 'upscaled':
        # 8-bit image scaled to 16-bit
        img_8 = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        img = img_8.astype(np.uint16) * 257 # 257 * 255 = 65535
        
    cv2.imwrite(path, img)
    return path

def main():
    print("--- 16-Bit Image Quality Validator PoC ---")
    
    # 1. Setup Test Files
    native_path = "test_native.png"
    upscaled_path = "test_upscaled.png"
    
    generate_test_image(native_path, mode='native')
    generate_test_image(upscaled_path, mode='upscaled')
    
    for path in [native_path, upscaled_path]:
        print(f"\nAnalyzing: {path}")
        img = load_image(path)
        
        bd = analyze_bit_depth(img)
        comb = detect_histogram_combing(img)
        
        print(f"  > Effective Depth: {bd['effective_depth']}-bit")
        print(f"  > Likely Upscaled 8-bit: {comb['likely_upscaled_8bit']} (Score: {comb['comb_strength_score']})")

    # Cleanup
    # os.remove(native_path)
    # os.remove(upscaled_path)

if __name__ == "__main__":
    main()
