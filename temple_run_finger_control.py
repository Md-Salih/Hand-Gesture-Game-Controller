import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)

# Control variables
last_jump_time = 0
last_duck_time = 0
jump_cooldown = 0.3  # Seconds between jumps
duck_cooldown = 0.5  # Seconds between ducks

# Hand position tracking
prev_y_positions = []
hand_stable_time = 0
HAND_STABLE_THRESHOLD = 0.3  # Seconds hand must be stable before detecting gesture

def detect_hand_gesture(hand_landmarks, frame_height):
    """Detect jump/duck gestures based on hand position"""
    # Get index finger tip (landmark 8)
    finger_tip_y = hand_landmarks.landmark[8].y
    
    # Convert to pixel coordinates
    finger_y = int(finger_tip_y * frame_height)
    
    # Store recent positions
    prev_y_positions.append(finger_y)
    if len(prev_y_positions) > 10:
        prev_y_positions.pop(0)
    
    # Calculate average position over recent frames
    if len(prev_y_positions) >= 5:
        avg_y = sum(prev_y_positions) / len(prev_y_positions)
        
        # Determine gesture based on hand height
        # Top third of screen = Jump
        # Bottom third of screen = Duck
        # Middle = No action
        
        if avg_y < frame_height * 0.35:  # Top area
            return "JUMP", avg_y
        elif avg_y > frame_height * 0.7:  # Bottom area
            return "DUCK", avg_y
    
    return "NONE", finger_y

def send_dino_command(gesture):
    """Send command to Chrome Dino game"""
    global last_jump_time, last_duck_time
    current_time = time.time()
    
    try:
        if gesture == "JUMP" and current_time - last_jump_time > jump_cooldown:
            pyautogui.press('space')
            print("🦕 JUMP!")
            last_jump_time = current_time
        elif gesture == "DUCK" and current_time - last_duck_time > duck_cooldown:
            pyautogui.press('down')
            print("🦕 DUCK!")
            last_duck_time = current_time
    except Exception as e:
        print(f"Error sending command: {e}")

print("🦕 Chrome Dino Game Finger Controller Started!")
print("🎮 How to play:")
print("   👆 Hold hand in TOP area = JUMP (Space)")
print("   👇 Hold hand in BOTTOM area = DUCK (Down arrow)")
print("   🖐️ Hold hand in MIDDLE area = No action")
print("\n🌐 To start Chrome Dino:")
print("   1. Open Chrome browser")
print("   2. Go offline or visit: chrome://dino/")
print("   3. Press Space to start the game")
print("   4. Keep the game window active!")
print("\n⏹️ Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw control zones
    # Jump zone (top)
    cv2.rectangle(frame, (0, 0), (w, int(h*0.35)), (0, 255, 0), 2)
    cv2.putText(frame, "JUMP ZONE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Duck zone (bottom)
    cv2.rectangle(frame, (0, int(h*0.7)), (w, h), (0, 0, 255), 2)
    cv2.putText(frame, "DUCK ZONE", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Neutral zone (middle)
    cv2.rectangle(frame, (0, int(h*0.35)), (w, int(h*0.7)), (255, 255, 0), 1)
    cv2.putText(frame, "NEUTRAL", (20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    current_gesture = "NONE"
    hand_y = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture
            current_gesture, hand_y = detect_hand_gesture(hand_landmarks, h)
            
            # Draw finger tip
            finger_tip = hand_landmarks.landmark[8]
            x = int(finger_tip.x * w)
            y = int(finger_tip.y * h)
            cv2.circle(frame, (x, y), 15, (255, 0, 255), -1)
            
            # Send command
            send_dino_command(current_gesture)
    else:
        # Clear position history when no hand detected
        prev_y_positions.clear()
    
    # Show status
    status_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
    status_text = "Hand Detected" if results.multi_hand_landmarks else "No Hand"
    cv2.putText(frame, status_text, (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Show current gesture
    gesture_color = (0, 255, 0) if current_gesture != "NONE" else (255, 255, 255)
    cv2.putText(frame, f"Action: {current_gesture}", (w-200, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
    
    # Show cooldown timers
    current_time = time.time()
    jump_cooldown_remaining = max(0, jump_cooldown - (current_time - last_jump_time))
    duck_cooldown_remaining = max(0, duck_cooldown - (current_time - last_duck_time))
    
    if jump_cooldown_remaining > 0:
        cv2.putText(frame, f"Jump cooldown: {jump_cooldown_remaining:.1f}s", 
                   (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    if duck_cooldown_remaining > 0:
        cv2.putText(frame, f"Duck cooldown: {duck_cooldown_remaining:.1f}s", 
                   (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Chrome Dino Controller", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Controller stopped!")