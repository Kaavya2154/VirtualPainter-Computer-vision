import cv2
import mediapipe as mp
import numpy as np
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
canvas = None

# Default values
draw_color = (0, 0, 255)  # red
brush_thickness = 7
eraser_thickness = 50
mode = "draw"
prev_x, prev_y = 0, 0

# Colors bar
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]
color_names = ["Red", "Green", "Blue", "Yellow", "Eraser"]

def draw_color_bar(img):
    bar_height = 60
    bar_width = 80
    for i, color in enumerate(colors):
        x = i * bar_width
        cv2.rectangle(img, (x, 0), (x + bar_width, bar_height), color, -1)
        cv2.putText(img, color_names[i], (x + 5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    frame = draw_color_bar(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            landmarks = hand_landmarks.landmark

            # Get index and middle finger tips
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            middle_x, middle_y = int(landmarks[12].x * w), int(landmarks[12].y * h)

            # Check if selection mode (two fingers up)
            fingers_up = landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y

            if fingers_up:
                prev_x, prev_y = 0, 0
                # Selecting color from top bar
                if index_y < 60:
                    idx = index_x // 80
                    if idx < len(colors):
                        if color_names[idx] == "Eraser":
                            mode = "erase"
                            draw_color = (0, 0, 0)
                        else:
                            mode = "draw"
                            draw_color = colors[idx]
                cv2.circle(frame, (index_x, index_y), 15, draw_color, cv2.FILLED)
            else:
                # Drawing mode
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y

                thickness = eraser_thickness if mode == "erase" else brush_thickness
                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), draw_color, thickness)
                prev_x, prev_y = index_x, index_y

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge drawing + camera
    blended = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.putText(blended, f"Mode: {mode}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(blended, "Press 'C' to Clear | 'S' to Save | 'Q' to Quit", (10, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    cv2.imshow("Virtual Painter", blended)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        canvas = np.zeros_like(frame)
    elif key & 0xFF == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"✅ Saved as {filename}")
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

