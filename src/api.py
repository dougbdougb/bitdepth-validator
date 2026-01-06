from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import uuid
from typing import Dict, Optional
import os

# Import our analysis modules
from .core import analyze_bit_depth, detect_histogram_combing
from .stress import run_stress_test
from .noise import run_noise_analysis

app = FastAPI(
    title="16-Bit Image Validator API",
    description="High-precision API for forensic image analysis and stress testing."
)

# --- In-Memory State Store ---
IMAGE_STORE: Dict[str, Dict] = {}

def numpy_to_png_bytes(img: np.ndarray) -> bytes:
    """
    Helper: Converts a numpy array to 8-bit PNG bytes for browser display.
    """
    if img.dtype == np.uint16:
        img_display = (img / 256).astype(np.uint8)
    else:
        img_display = img.astype(np.uint8)

    success, encoded_image = cv2.imencode(".png", img_display)
    if not success:
        raise ValueError("Failed to encode image for display.")
    return encoded_image.tobytes()

# --- Endpoints ---

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        session_id = str(uuid.uuid4())
        IMAGE_STORE[session_id] = {
            "original": img,
            "filename": file.filename
        }
        
        return {
            "session_id": session_id, 
            "message": "Image loaded into memory.",
            "resolution": f"{img.shape[1]}x{img.shape[0]}",
            "dtype": str(img.dtype)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/authenticity/{session_id}")
def get_authenticity(session_id: str):
    if session_id not in IMAGE_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    img = IMAGE_STORE[session_id]["original"]
    return {
        "bit_depth": analyze_bit_depth(img),
        "histogram": detect_histogram_combing(img)
    }

@app.get("/analyze/noise/{session_id}")
def get_noise(session_id: str):
    if session_id not in IMAGE_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    img = IMAGE_STORE[session_id]["original"]
    return run_noise_analysis(img)

@app.get("/analyze/stress/{session_id}")
def get_stress_metrics(session_id: str, intensity: float = Query(15.0, ge=1.0, le=50.0)):
    if session_id not in IMAGE_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    img = IMAGE_STORE[session_id]["original"]
    result = run_stress_test(img, intensity)
    
    # Store visuals in session
    IMAGE_STORE[session_id]["stress_preview"] = result.pop("stressed_image_preview")
    IMAGE_STORE[session_id]["banding_mask"] = result.pop("heatmap_mask")
    
    return result

@app.get("/visualize/banding/{session_id}")
def get_banding_overlay(session_id: str):
    """
    Returns the original image with banding artifacts highlighted in RED.
    """
    if session_id not in IMAGE_STORE or "banding_mask" not in IMAGE_STORE[session_id]:
        raise HTTPException(status_code=404, detail="Stress test has not been run.")
        
    original = IMAGE_STORE[session_id]["original"]
    mask = IMAGE_STORE[session_id]["banding_mask"]
    
    # Convert to RGB for color overlay
    if len(original.shape) == 2:
        vis = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        vis = original.copy()
        
    # Paint banding areas Red (BGR format: 0, 0, 65535 for 16-bit)
    vis[mask] = [0, 0, 65535]
    
    return Response(content=numpy_to_png_bytes(vis), media_type="image/png")

@app.get("/visualize/stress/{session_id}")
def get_stress_visualization(session_id: str):
    if session_id not in IMAGE_STORE or "stress_preview" not in IMAGE_STORE[session_id]:
        raise HTTPException(status_code=404, detail="Stress test has not been run for this session.")
    img = IMAGE_STORE[session_id]["stress_preview"]
    return Response(content=numpy_to_png_bytes(img), media_type="image/png")

# --- Static File Serving ---

# Create absolute path for static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# --- Visualization Endpoints ---

@app.get("/visualize/original/{session_id}")
def get_original_visualization(session_id: str):
    if session_id not in IMAGE_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    img = IMAGE_STORE[session_id]["original"]
    return Response(content=numpy_to_png_bytes(img), media_type="image/png")

@app.get("/visualize/noise_residual/{session_id}")
def get_noise_residual_visualization(session_id: str, boost: float = Query(1.0, ge=1.0, le=20.0)):
    if session_id not in IMAGE_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    img = IMAGE_STORE[session_id]["original"]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    from .noise import extract_noise_plane
    residual = extract_noise_plane(img)
    
    # Normalize to 0-255 around a 128 mid-point
    # (residual is difference, so it can be negative)
    # We apply the boost here
    res_boosted = (residual * boost) + 128
    res_clipped = np.clip(res_boosted, 0, 255).astype(np.uint8)
    
    return Response(content=numpy_to_png_bytes(res_clipped), media_type="image/png")

@app.get("/visualize/fft_spectrum/{session_id}")
def get_fft_spectrum_visualization(session_id: str, boost: float = Query(1.0, ge=1.0, le=5.0)):
    if session_id not in IMAGE_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    img = IMAGE_STORE[session_id]["original"]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    from scipy.fft import fft2, fftshift
    from .noise import extract_noise_plane
    noise = extract_noise_plane(img)
    f = fft2(noise)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    
    # Normalize with optional boost
    spec_min, spec_max = np.min(magnitude_spectrum), np.max(magnitude_spectrum)
    spec_norm = ((magnitude_spectrum - spec_min) / (spec_max - spec_min + 1e-6) * 255)
    
    # Apply boost on the normalized spectrum to pop spikes
    spec_boosted = np.clip(spec_norm * boost, 0, 255).astype(np.uint8)
    spec_color = cv2.applyColorMap(spec_boosted, cv2.COLORMAP_VIRIDIS)
    
    return Response(content=numpy_to_png_bytes(spec_color), media_type="image/png")

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in IMAGE_STORE:
        del IMAGE_STORE[session_id]
        return {"message": "Session cleared."}
    return {"message": "Session not found."}
