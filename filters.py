import cv2
import numpy as np

# Dictionary of filters using OpenCV built-in functions
def apply_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_sepia(frame):
    # Create sepia filter matrix (3x3 for BGR channels)
    kernel = np.array([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    # Clip values to stay in valid range [0,255]
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)


def apply_cartoon(frame):
    # Edge-preserving filter + edge detection
    color = cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(edges, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 9)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_colormap(frame):
    return cv2.applyColorMap(frame, cv2.COLORMAP_JET)

def apply_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
def apply_glow(frame): #specially added by me
    smoothed = cv2.bilateralFilter(frame, d=15, sigmaColor=75, sigmaSpace=75)
    brightens = cv2.convertScaleAbs(smoothed, alpha=1.1, beta=20)
    blur = cv2.GaussianBlur(brightens,(21,21),0)
    glow = cv2.addWeighted(brightens, 0.8, blur, 0.2, 0)
    return glow
def apply_pink(frame): # THIS ONE TOO
    return cv2.applyColorMap(frame, cv2.COLORMAP_PINK)

filters = {
    '1': apply_gray,
    '2': apply_sepia,
    '3': apply_cartoon,
    '4': apply_colormap,
    '5': apply_sketch,
    '6': apply_glow,
    '7': apply_pink
}

cap = cv2.VideoCapture(0)
current_filter = None

print("Press keys 1–7 to change filters, 's' to save, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if current_filter:
        frame = current_filter(frame)

    cv2.imshow("OpenCV Filters", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in filters:
        current_filter = filters[chr(key)]
        print(f"Filter switched to {chr(key)}")
    elif key == ord('s'):
        cv2.imwrite("snapshot.png", frame)
        print("Snapshot saved!")

cap.release()
cv2.destroyAllWindows()

