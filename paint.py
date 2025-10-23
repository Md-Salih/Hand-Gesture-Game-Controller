import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

drawing = False  # track if mouse is pressed
smooth_buffer = deque(maxlen=5)  # buffer for smoothing

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Index finger tip (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            index_x = int(index_finger.x * w)
            index_y = int(index_finger.y * h)

            # Thumb tip (landmark 4)
            thumb = hand_landmarks.landmark[4]
            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Map webcam coords to screen coords
            screen_x = np.interp(index_x, (0, w), (0, screen_w))
            screen_y = np.interp(index_y, (0, h), (0, screen_h))

            # Add to smoothing buffer
            smooth_buffer.append((screen_x, screen_y))
            avg_x = np.mean([p[0] for p in smooth_buffer])
            avg_y = np.mean([p[1] for p in smooth_buffer])

            # Move mouse smoothly
            pyautogui.moveTo(avg_x, avg_y)

            # Check pinch distance to draw
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if distance < 40 and not drawing:
                pyautogui.mouseDown()
                drawing = True
            elif distance >= 40 and drawing:
                pyautogui.mouseUp()
                drawing = False

    cv2.imshow("Paint Controller", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
