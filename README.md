# 16-Bit Image Quality Validator (Forensic Inspector)

> [!IMPORTANT]
> **VIBE CODED EXPERIMENT**: This application is a "VIBE Coded" proof-of-concept. It was built rapidly to explore forensic methods for evaluating 16-bit imagery. It is NOT a production-ready tool, but rather a research playground for documenting and sharing ideas on bit-depth validation.

## Overiew
This tool provides a two-pronged approach to identifying "fake" 16-bit data (upscaled 8-bit) and detecting banding artifacts in high-precision gradients (e.g., skies, renders).

1.  **Automated CLI**: Rapid inspection of files for pipeline integration.
2.  **Web Dashboard**: Visual forensic playground with contrast stressing, FFT spectrum analysis, and noise residual isolation.

## Key Forensic Features
- **Heuristic 6.1 (Texture-Aware Banding)**: Uses Laplacian masking to ignore busy areas (mountains, trees) and focus only on smooth gradients where banding occurs.
- **Bit-Depth Discovery**: Detects effective bit-depth vs. container depth and checks for LSB padding or "histogram combing."
- **Nature Analysis (FFT)**: Identifies geometric patterns (stars/crosses) used to hide upscaling via dither or AI resampling.
- **Residual Mode**: High-pass filtering to isolate the "noise floor"—distinguishes between natural grain and synthetic noise.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run the Web Dashboard
```bash
python -m uvicorn src.api:app --reload
```
Open `http://localhost:8000` to access the forensic UI.

### Run the CLI
```bash
# Core metadata and upscaling check
python -m src.cli inspect path/to/image.png

# Contrast stress test (banding detection)
python -m src.cli stress path/to/image.png

# Noise profile and dithering test
python -m src.cli noise path/to/image.png
```

## Forensic Interpretations (Quick Guide)
- **Random Sand (Residual)**: Natural high-quality sensor grain.
- **Geometric Grids (Residual)**: Synthetic dithering used to mask bit-depth defects.
- **Solid Flat Grey (Residual)**: Clean for Digital Art/Animation; suspicious "crushed" texture for Photos.
- **FFT Cross/Stars (Spectrum)**: Mathematical artifacts from upscaling, compression, or AI processing.
- **Banding Heatmap (Stress)**: Red highlights indicate where 16-bit gradients have been torn into discrete 8-bit steps.

---
*Created as a "Vibe Coding" exploration for Advanced Image Quality Analysis.*
