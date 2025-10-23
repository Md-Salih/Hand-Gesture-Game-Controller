import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Track previous finger position
prev_x, prev_y = None, None
gesture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index finger tip = landmark 8
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y

                # 0.5s cooldown so it doesn't spam
                if time.time() - gesture_time > 0.5:
                    if abs(dx) > abs(dy):  # Horizontal movement
                        if dx > 40:
                            pyautogui.press('right')
                            print("Right swipe detected")
                            gesture_time = time.time()
                        elif dx < -40:
                            pyautogui.press('left')
                            print("Left swipe detected")
                            gesture_time = time.time()
                    else:  # Vertical movement
                        if dy < -40:
                            pyautogui.press('up')
                            print("Up swipe detected (Jump)")
                            gesture_time = time.time()
                        elif dy > 40:
                            pyautogui.press('down')
                            print("Down swipe detected (Slide)")
                            gesture_time = time.time()

            prev_x, prev_y = x, y

    cv2.imshow("Temple Run Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
