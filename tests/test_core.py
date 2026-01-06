import pytest
import numpy as np
import cv2
import os
from src.core import analyze_bit_depth, detect_histogram_combing

def test_analyze_bit_depth_native():
    # 16-bit noise should have 16 bits active
    img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    results = analyze_bit_depth(img)
    assert results['effective_depth'] == 16
    assert results['is_padded'] == False

def test_analyze_bit_depth_12bit_in_16bit():
    # 12-bit data shifted up by 4 bits
    img_12 = np.random.randint(0, 4095, (100, 100), dtype=np.uint16)
    img_16 = img_12 << 4
    results = analyze_bit_depth(img_16)
    assert results['effective_depth'] == 16
    assert results['lowest_active_bit'] == 4
    assert results['is_padded'] == True

def test_detect_histogram_combing_native():
    # Uniform noise should not have combing
    img = np.random.randint(0, 65535, (200, 200), dtype=np.uint16)
    results = detect_histogram_combing(img)
    assert results['likely_upscaled_8bit'] == False

def test_detect_histogram_combing_upscaled():
    # 8-bit data upscaled
    img_8 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    img_16 = img_8.astype(np.uint16) * 257
    results = detect_histogram_combing(img_16)
    assert results['likely_upscaled_8bit'] == True
    assert results['comb_strength_score'] > 50
