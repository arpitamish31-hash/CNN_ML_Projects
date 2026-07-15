# === Step 1: Import Required Libraries ===
import cv2                     # OpenCV for webcam and image processing
import mediapipe as mp         # MediaPipe for hand tracking
import pyautogui               # Keyboard/mouse automation (controls PowerPoint)
import numpy as np             # Numerical arrays (for annotation canvas)
import time                    # Timing control (delays between actions)
import os                      # File system operations
from pptx import Presentation  # PowerPoint file handling

# === Step 2: Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,   # Confidence threshold for detection
    min_tracking_confidence=0.7     # Confidence threshold for tracking
)

# === Step 3: Setup Annotation Canvas and Flags ===
canvas = None        # Will hold drawings/annotations
drawing = False      # Flag for drawing mode
laser_mode = False   # Flag for laser pointer mode

# === Step 4: Webcam Capture ===
cap = cv2.VideoCapture(0)  # 0 = default webcam

# === Step 5: Load PowerPoint File (Optional) ===
ppt_path = "example.pptx"
if os.path.exists(ppt_path):
    prs = Presentation(ppt_path)
    print(f"Loaded presentation with {len(prs.slides)} slides.")
else:
    print("PowerPoint file not found! Running without PPT integration.")

# === Step 6: Helper Function to Detect Fingers Up ===
def fingers_up(hand_landmarks):
    """
    Returns a list of 5 values (1=up, 0=down) for Thumb, Index, Middle, Ring, Pinky.
    """
    tips = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
    fingers = []

    # Thumb: compare x-coordinates
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: compare y-coordinates
    for id in range(1, 5):
        if hand_landmarks.landmark[tips[id]].y < hand_landmarks.landmark[tips[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# === Step 7: Main Loop for Gesture Detection ===
print("Starting webcam... Use gestures to control PowerPoint.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Initialize canvas once
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # === Step 8: Process Hand Landmarks ===
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger states
            fingers = fingers_up(hand_landmarks)

            # === Step 9: Gesture Mappings ===
            # Next Slide (Index finger up)
            if fingers == [0,1,0,0,0]:
                pyautogui.press("right")
                time.sleep(0.5)

            # Previous Slide (Index + Middle up)
            elif fingers == [0,1,1,0,0]:
                pyautogui.press("left")
                time.sleep(0.5)

            # Start Slideshow (Thumb + Pinky up)
            elif fingers == [1,0,0,0,1]:
                pyautogui.press("f5")
                time.sleep(1)

            # End Slideshow (Fist: all fingers down)
            elif fingers == [0,0,0,0,0]:
                pyautogui.press("esc")
                time.sleep(1)

            # Laser Pointer Mode (Thumb up)
            elif fingers == [1,0,0,0,0]:
                laser_mode = True
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                cv2.circle(frame, (x,y), 15, (0,0,255), -1)
            else:
                laser_mode = False

            # Draw/Annotate (Thumb + Index up)
            if fingers == [1,1,0,0,0]:
                drawing = True
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                cv2.circle(canvas, (x,y), 10, (0,255,0), -1)
            else:
                drawing = False

            # Clear Annotations (All fingers up)
            if fingers == [1,1,1,1,1]:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # === Step 10: Merge Canvas with Frame ===
    frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Display output
    cv2.imshow("Gesture Presentation", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Step 11: Release Resources ===
cap.release()
cv2.destroyAllWindows()
