import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class GestureConfig:
    """Configuration for gesture detection"""
    max_hands: int = 1
    gesture_threshold: float = 40.0
    cooldown_time: float = 0.8
    smoothing_factor: float = 0.7
    confidence_threshold: float = 0.7

class GestureController:
    def __init__(self, config: GestureConfig = None):
        self.config = config or GestureConfig()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=self.config.confidence_threshold,
            min_tracking_confidence=self.config.confidence_threshold
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Tracking variables
        self.prev_positions = []
        self.last_gesture_time = 0
        self.gesture_history = []
        
        # Smoothing buffer
        self.position_buffer = []
        self.buffer_size = 5
        
        # Safety settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
    def smooth_position(self, x: int, y: int) -> Tuple[int, int]:
        """Apply smoothing to reduce jitter"""
        self.position_buffer.append((x, y))
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
            
        # Weighted average with more weight on recent positions
        weights = np.linspace(0.5, 1.0, len(self.position_buffer))
        weights /= weights.sum()
        
        smooth_x = sum(pos[0] * w for pos, w in zip(self.position_buffer, weights))
        smooth_y = sum(pos[1] * w for pos, w in zip(self.position_buffer, weights))
        
        return int(smooth_x), int(smooth_y)
    
    def detect_gesture(self, current_pos: Tuple[int, int]) -> Optional[str]:
        """Detect gesture based on finger movement"""
        if not self.prev_positions:
            return None
            
        prev_pos = self.prev_positions[-1]
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        current_time = time.time()
        
        # Check if movement is significant and cooldown has passed
        if (distance > self.config.gesture_threshold and 
            (current_time - self.last_gesture_time) > self.config.cooldown_time):
            
            # Determine primary direction
            if abs(dx) > abs(dy):
                gesture = "swipe_right" if dx > 0 else "swipe_left"
            else:
                gesture = "swipe_down" if dy > 0 else "swipe_up"
                
            self.last_gesture_time = current_time
            self.gesture_history.append((gesture, current_time))
            
            # Keep only recent history
            self.gesture_history = [(g, t) for g, t in self.gesture_history 
                                  if current_time - t < 10]
            
            return gesture
        
        return None
    
    def execute_gesture_action(self, gesture: str) -> None:
        """Execute the corresponding action for detected gesture"""
        actions = {
            "swipe_right": lambda: pyautogui.press('right'),
            "swipe_left": lambda: pyautogui.press('left'),
            "swipe_up": lambda: pyautogui.press('up'),
            "swipe_down": lambda: pyautogui.press('down')
        }
        
        try:
            if gesture in actions:
                actions[gesture]()
                print(f"✓ Executed: {gesture.replace('_', ' ').title()}")
        except Exception as e:
            print(f"✗ Error executing gesture {gesture}: {e}")
    
    def draw_info_overlay(self, img: np.ndarray, gesture: Optional[str] = None) -> np.ndarray:
        """Draw information overlay on the image"""
        h, w = img.shape[:2]
        
        # Create semi-transparent overlay
        overlay = img.copy()
        
        # Draw status information
        status_text = "Hand Gesture Control Active"
        cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Show recent gesture
        if gesture:
            gesture_text = f"Gesture: {gesture.replace('_', ' ').title()}"
            cv2.putText(overlay, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
        
        # Show gesture history count
        history_text = f"Gestures detected: {len(self.gesture_history)}"
        cv2.putText(overlay, history_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "ESC: Quit",
            "R: Reset tracking",
            "Space: Pause/Resume"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(overlay, instruction, (w - 200, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return overlay
    
    def reset_tracking(self) -> None:
        """Reset all tracking variables"""
        self.prev_positions.clear()
        self.position_buffer.clear()
        self.gesture_history.clear()
        print("🔄 Tracking reset")
    
    def process_frame(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Process a single frame and detect gestures"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        h, w = img.shape[:2]
        detected_gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip (landmark 8)
                landmark = hand_landmarks.landmark[8]
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                # Apply smoothing
                smooth_x, smooth_y = self.smooth_position(x, y)
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Draw finger tip with enhanced visual
                cv2.circle(img, (smooth_x, smooth_y), 10, (255, 0, 0), -1)
                cv2.circle(img, (smooth_x, smooth_y), 15, (255, 255, 255), 2)
                
                # Detect gesture
                detected_gesture = self.detect_gesture((smooth_x, smooth_y))
                
                # Update position history
                self.prev_positions.append((smooth_x, smooth_y))
                if len(self.prev_positions) > 10:
                    self.prev_positions.pop(0)
                
                # Draw movement trail
                if len(self.prev_positions) > 1:
                    for i in range(1, len(self.prev_positions)):
                        cv2.line(img, self.prev_positions[i-1], self.prev_positions[i], 
                                (0, 255, 255), 2)
        
        return img, detected_gesture

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced Gesture Control System")
    print("📋 Instructions:")
    print("   - Move your index finger to control")
    print("   - ESC: Quit")
    print("   - R: Reset tracking")
    print("   - Space: Pause/Resume")
    print("-" * 50)
    
    # Initialize controller
    config = GestureConfig(
        gesture_threshold=35.0,
        cooldown_time=0.8,
        confidence_threshold=0.7
    )
    
    controller = GestureController(config)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    paused = False
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                print("❌ Failed to read from camera")
                break
            
            # Flip image horizontally for mirror effect
            img = cv2.flip(img, 1)
            
            frame_count += 1
            
            if not paused:
                # Process frame for gesture detection
                img, detected_gesture = controller.process_frame(img)
                
                # Execute gesture action
                if detected_gesture:
                    controller.execute_gesture_action(detected_gesture)
            else:
                # Show paused indicator
                cv2.putText(img, "PAUSED", (img.shape[1]//2 - 60, img.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Draw information overlay
            img = controller.draw_info_overlay(img, detected_gesture if not paused else None)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            cv2.imshow("Enhanced Gesture Control", img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                controller.reset_tracking()
            elif key == ord(' '):  # Space
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        controller.hands.close()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()