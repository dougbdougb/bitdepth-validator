import cv2
import numpy as np

def get_profile(filepath, label):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None: return
    
    # Take a vertical slice through the sky
    # Let's say x=500, y from 0 to 300
    slice_line = img[0:300, 500, 0] # Blue channel
    
    print(f"\n[{label}] Top 20 pixels values:")
    print(slice_line[:20])
    
    # Calculate jumps
    jumps = np.diff(slice_line.astype(np.int32))
    print(f"[{label}] Unique jumps in sky (slice): {np.unique(jumps)}")
    
    # Plateau lengths
    plateaus = []
    current_p = 1
    for i in range(1, len(slice_line)):
        if slice_line[i] == slice_line[i-1]:
            current_p += 1
        else:
            plateaus.append(current_p)
            current_p = 1
    print(f"[{label}] Max plateau length: {max(plateaus) if plateaus else 0}")
    print(f"[{label}] Avg plateau length: {np.mean(plateaus) if plateaus else 0:.2f}")

file_a = "/Users/doug/.gemini/antigravity/brain/7e0980d3-18b1-499d-a8c1-d11a6cd64350/uploaded_image_0_1767660112603.png"
file_b = "/Users/doug/.gemini/antigravity/brain/7e0980d3-18b1-499d-a8c1-d11a6cd64350/uploaded_image_1_1767660112603.png"

get_profile(file_a, "FILE_A")
get_profile(file_b, "FILE_B")
