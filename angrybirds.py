import cv2
import mediapipe as mp
import pyautogui
import time

# Setup mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

dragging = False
start_pos = None

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, c = img.shape

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]

        # Finger tip landmark (index finger tip = 8)
        x = int(handLms.landmark[8].x * w)
        y = int(handLms.landmark[8].y * h)

        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        # Detect if index finger is "down" or "up"
        # We can check distance between tip (8) and pip joint (6) of index finger to determine if finger is bent (down)
        tip = handLms.landmark[8]
        pip = handLms.landmark[6]
        finger_folded = tip.y > pip.y  # if tip below pip in y axis => finger folded (down)

        screen_x, screen_y = pyautogui.size()

        # Map webcam coords to screen coords (flip x)
        mouse_x = screen_x - int(handLms.landmark[8].x * screen_x)
        mouse_y = int(handLms.landmark[8].y * screen_y)

        if not dragging and finger_folded:
            # Start dragging (mouse down)
            dragging = True
            start_pos = (mouse_x, mouse_y)
            pyautogui.mouseDown(x=mouse_x, y=mouse_y)
            print("Drag started at", start_pos)

        elif dragging and not finger_folded:
            # Finger released => mouse up
            dragging = False
            pyautogui.mouseUp(x=mouse_x, y=mouse_y)
            print("Drag released at", (mouse_x, mouse_y))
            start_pos = None

        elif dragging:
            # Move mouse while dragging
            pyautogui.moveTo(mouse_x, mouse_y)
            cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)

    else:
        # No hand detected, release drag if dragging
        if dragging:
            dragging = False
            pyautogui.mouseUp()
            print("Drag cancelled")

    cv2.imshow("Angry Birds Finger Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
