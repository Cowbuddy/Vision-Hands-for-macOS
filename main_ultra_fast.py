#!/usr/bin/env python3
"""
ULTRA FAST Hand Tracking System - Apple Vision Pro Style Controls

Optimized for maximum performance with Apple Vision Pro-style intuitive controls.
Right-hand cursor control with sensitivity adjustment and smooth tracking.
Features: EMA smoothing, conditional updates, low-latency processing.
"""
import os
import cv2
import time
import numpy as np
import threading
from typing import Optional, Tuple, Dict, Any
from collections import deque

# Import MediaPipe
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode
)

# Import optimized components
from src.models import HandInfo, HandLandmarks
from src.hand_analyzer import HandAnalyzer
from src.enhanced_gesture_recognition import StableGestureRecognizer
from src.system_controller import SystemController

# macOS native cursor control for ultra-fast response
try:
    import Quartz
    MACOS_NATIVE = True
    print("✅ macOS native cursor APIs loaded")
except ImportError:
    import pyautogui
    MACOS_NATIVE = False
    print("⚠️  Using pyautogui (install pyobjc for native speed)")

# Configure for maximum performance
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Constants
MODEL_PATH = "hand_landmarker.task"


class NativeCursorController:
    """Ultra-fast native macOS cursor control"""
    
    def __init__(self):
        self.last_position = None
        self.is_clicking = False
        self.click_held = False
        
    def move_cursor(self, x: float, y: float):
        """Move cursor using native macOS APIs for minimum latency"""
        if MACOS_NATIVE:
            # Use Quartz for ultra-fast cursor movement
            Quartz.CGWarpMouseCursorPosition((x, y))
        else:
            # Fallback to pyautogui
            pyautogui.moveTo(x, y, duration=0)
        
        self.last_position = (x, y)
    
    def click(self):
        """Perform click"""
        if MACOS_NATIVE:
            # Native macOS click
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, (0, 0), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, (0, 0), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        else:
            pyautogui.click()
    
    def start_drag(self):
        """Start drag operation"""
        if MACOS_NATIVE and not self.click_held:
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, (0, 0), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            self.click_held = True
        elif not MACOS_NATIVE:
            pyautogui.mouseDown()
            self.click_held = True
    
    def end_drag(self):
        """End drag operation"""
        if MACOS_NATIVE and self.click_held:
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, (0, 0), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            self.click_held = False
        elif not MACOS_NATIVE and self.click_held:
            pyautogui.mouseUp()
            self.click_held = False
    
    def right_click(self):
        """Perform right click"""
        if MACOS_NATIVE:
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventRightMouseDown, (0, 0), Quartz.kCGMouseButtonRight)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventRightMouseUp, (0, 0), Quartz.kCGMouseButtonRight)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        else:
            pyautogui.rightClick()


class EMASmoothing:
    """Exponential Moving Average for ultra-smooth cursor movement"""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.smoothed_x = None
        self.smoothed_y = None
        
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update with new position and return smoothed coordinates"""
        if self.smoothed_x is None:
            self.smoothed_x = x
            self.smoothed_y = y
        else:
            self.smoothed_x = self.alpha * x + (1 - self.alpha) * self.smoothed_x
            self.smoothed_y = self.alpha * y + (1 - self.alpha) * self.smoothed_y
        
        return self.smoothed_x, self.smoothed_y
    
    def set_sensitivity(self, sensitivity: float):
        """Adjust sensitivity (0.1 = very smooth, 0.9 = very responsive)"""
        self.alpha = max(0.1, min(0.9, sensitivity))
    
    def reset(self):
        """Reset smoothing state"""
        self.smoothed_x = None
        self.smoothed_y = None


def create_ultra_fast_landmarker():
    """Create ultra-optimized hand landmarker"""
    print("🚀 Initializing ULTRA FAST hand tracking...")
    print("⚡ Apple Vision Pro style controls")
    
    base_opts = BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=BaseOptions.Delegate.CPU  # CPU is faster on M4 Pro for this task
    )
    
    options = HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,  # Lower for speed
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return HandLandmarker.create_from_options(options)


class UltraFastHandTracker:
    """Ultra-fast hand tracking optimized for mouse replacement"""
    
    def __init__(self):
        # Core components
        self.hand_landmarker = create_ultra_fast_landmarker()
        self.hand_analyzer = HandAnalyzer()
        self.gesture_recognizer = StableGestureRecognizer(history_size=8)  # Smaller for speed
        self.cursor_controller = NativeCursorController()
        self.system_controller = SystemController()
        
        # Performance settings
        self.process_every_n_frames = 1  # Process every frame initially
        self.frame_count = 0
        self.fps_target = 60
        self.fps_actual = 0
        
        # Cursor settings
        self.cursor_enabled = True
        self.sensitivity = 0.4  # Default sensitivity
        self.movement_threshold = 3  # Minimum pixels to move cursor
        self.ema_smoother = EMASmoothing(alpha=self.sensitivity)
        
        # Screen dimensions
        if MACOS_NATIVE:
            main_display = Quartz.CGMainDisplayID()
            self.screen_width = Quartz.CGDisplayPixelsWide(main_display)
            self.screen_height = Quartz.CGDisplayPixelsHigh(main_display)
        else:
            self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture state tracking
        self.last_cursor_pos = None
        self.gesture_states = {
            'pinch_held': False,
            'drag_active': False,
            'right_click_ready': False,
            'scroll_active': False,
            'last_scroll_time': 0
        }
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.last_fps_update = time.time()
        
        print("✅ Ultra-fast system ready!")
        print("🎯 RIGHT HAND CONTROLS:")
        print("   • Index finger extended: Move cursor")
        print("   • Pinch (thumb+index): Click")
        print("   • Pinch held: Drag")
        print("   • Peace sign: Right click")
        print("   • 3 fingers up: Scroll up")
        print("   • 4 fingers up: Scroll down")
        print("   • Fist: Pause cursor")
        print("⚙️  SENSITIVITY: Press 1-9 to adjust (1=slow, 9=fast)")
    
    def adjust_sensitivity(self, level: int):
        """Adjust cursor sensitivity (1-9)"""
        if 1 <= level <= 9:
            self.sensitivity = 0.1 + (level - 1) * 0.1  # 0.1 to 0.9
            self.ema_smoother.set_sensitivity(self.sensitivity)
            print(f"🎯 Sensitivity set to {level}/9 (alpha={self.sensitivity:.1f})")
    
    def should_skip_frame(self) -> bool:
        """Dynamic frame skipping based on performance"""
        if len(self.frame_times) < 10:
            return False
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 60
        
        if current_fps < self.fps_target * 0.8:  # If below 80% of target
            self.process_every_n_frames = min(3, self.process_every_n_frames + 1)
        elif current_fps > self.fps_target * 0.95:  # If above 95% of target
            self.process_every_n_frames = max(1, self.process_every_n_frames - 1)
        
        return (self.frame_count % self.process_every_n_frames) != 0
    
    def process_frame_ultra_fast(self, frame: np.ndarray) -> np.ndarray:
        """Ultra-optimized frame processing"""
        start_time = time.time()
        self.frame_count += 1
        
        # Dynamic frame skipping for performance
        if self.should_skip_frame():
            return self._draw_minimal_ui(frame)
        
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        timestamp_ms = int(self.frame_count * 16.67)  # ~60fps timestamps
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Process hands for cursor control
        if result.hand_landmarks and result.handedness:
            self._process_hands_for_cursor(
                result.hand_landmarks, result.handedness, frame_width, frame_height
            )
        
        # Draw UI
        annotated_frame = self._draw_ultra_fast_ui(frame, result)
        
        # Update performance metrics
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        if time.time() - self.last_fps_update >= 1.0:
            self.fps_actual = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            self.last_fps_update = time.time()
        
        return annotated_frame
    
    def _process_hands_for_cursor(self, hand_landmarks_list, handedness_list, frame_width, frame_height):
        """Process hands specifically for cursor control"""
        right_hand = None
        left_hand = None
        
        # Identify hands
        for landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            hand_info = self.hand_analyzer.analyze_hand(landmarks, handedness, frame_width, frame_height)
            
            if hand_info.hand_type == "Right":
                right_hand = hand_info
            else:
                left_hand = hand_info
        
        # Process right hand for cursor control (primary)
        if right_hand and self.cursor_enabled:
            self._handle_right_hand_cursor(right_hand, frame_width, frame_height)
        
        # Process left hand for commands (secondary)
        if left_hand:
            self._handle_left_hand_commands(left_hand)
    
    def _handle_right_hand_cursor(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Handle right hand cursor control with Apple Vision style"""
        # Get gesture
        gesture = self.gesture_recognizer.recognize_gesture(
            hand_info.fingers, hand_info.is_fist, hand_info.is_pinching
        )
        
        # Count extended fingers for scrolling
        extended_fingers = sum(1 for finger in hand_info.fingers.values() if finger.is_extended)
        current_time = time.time()
        
        if gesture == "point":
            # Index finger extended - move cursor
            self._move_cursor_with_index_finger(hand_info, frame_width, frame_height)
            
        elif gesture == "pinch":
            # Pinch gesture - click or drag
            if not self.gesture_states['pinch_held']:
                # Start of pinch - click or start drag
                self.cursor_controller.click()
                self.gesture_states['pinch_held'] = True
                print("🖱️ Click")
            else:
                # Pinch held - drag mode
                if not self.gesture_states['drag_active']:
                    self.cursor_controller.start_drag()
                    self.gesture_states['drag_active'] = True
                    print("🖱️ Drag started")
                
                # Continue moving while dragging
                self._move_cursor_with_pinch(hand_info, frame_width, frame_height)
                
        elif gesture == "peace":
            # Peace sign - right click
            if not self.gesture_states['right_click_ready']:
                self.cursor_controller.right_click()
                self.gesture_states['right_click_ready'] = True
                print("🖱️ Right click")
                
        elif extended_fingers == 3 and current_time - self.gesture_states['last_scroll_time'] > 0.3:
            # 3 fingers up - scroll up
            self.system_controller.scroll("up", 3)
            self.gesture_states['last_scroll_time'] = current_time
            print("📜 Scroll up")
            
        elif extended_fingers == 4 and current_time - self.gesture_states['last_scroll_time'] > 0.3:
            # 4 fingers up - scroll down  
            self.system_controller.scroll("down", 3)
            self.gesture_states['last_scroll_time'] = current_time
            print("📜 Scroll down")
                
        elif gesture == "fist":
            # Fist - pause cursor movement
            pass
            
        else:
            # Reset gesture states when hand is open or gesture changes
            if self.gesture_states['drag_active']:
                self.cursor_controller.end_drag()
                self.gesture_states['drag_active'] = False
                print("🖱️ Drag ended")
            
            self.gesture_states['pinch_held'] = False
            self.gesture_states['right_click_ready'] = False
    
    def _move_cursor_with_index_finger(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Move cursor using index finger tip position"""
        if not hand_info.fingers["index"].is_extended:
            return
        
        # Get index finger tip position
        index_tip = hand_info.fingers["index"].tip_position
        
        # Convert to screen coordinates
        screen_x = (index_tip[0] / frame_width) * self.screen_width
        screen_y = (index_tip[1] / frame_height) * self.screen_height
        
        # Apply EMA smoothing
        smoothed_x, smoothed_y = self.ema_smoother.update(screen_x, screen_y)
        
        # Only move if significant change
        if self.last_cursor_pos is None:
            self.cursor_controller.move_cursor(smoothed_x, smoothed_y)
            self.last_cursor_pos = (smoothed_x, smoothed_y)
        else:
            dx = abs(smoothed_x - self.last_cursor_pos[0])
            dy = abs(smoothed_y - self.last_cursor_pos[1])
            
            if dx > self.movement_threshold or dy > self.movement_threshold:
                self.cursor_controller.move_cursor(smoothed_x, smoothed_y)
                self.last_cursor_pos = (smoothed_x, smoothed_y)
    
    def _move_cursor_with_pinch(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Move cursor during pinch/drag operation"""
        # Use center of pinch (between thumb and index) for drag movement
        thumb_tip = hand_info.fingers["thumb"].tip_position
        index_tip = hand_info.fingers["index"].tip_position
        
        # Calculate pinch center
        pinch_center_x = (thumb_tip[0] + index_tip[0]) / 2
        pinch_center_y = (thumb_tip[1] + index_tip[1]) / 2
        
        # Convert to screen coordinates
        screen_x = (pinch_center_x / frame_width) * self.screen_width
        screen_y = (pinch_center_y / frame_height) * self.screen_height
        
        # Apply lighter smoothing for drag (more responsive)
        light_smoother = EMASmoothing(alpha=min(0.8, self.sensitivity + 0.3))
        smoothed_x, smoothed_y = light_smoother.update(screen_x, screen_y)
        
        # Move cursor (no threshold check during drag for precision)
        self.cursor_controller.move_cursor(smoothed_x, smoothed_y)
        self.last_cursor_pos = (smoothed_x, smoothed_y)
    
    def _handle_left_hand_commands(self, hand_info: HandInfo):
        """Handle left hand commands for system control"""
        gesture = self.gesture_recognizer.recognize_gesture(
            hand_info.fingers, hand_info.is_fist, hand_info.is_pinching
        )
        
        # Left hand can control cursor enable/disable
        if gesture == "fist":
            self.cursor_enabled = False
            print("⏸️ Cursor disabled")
        elif gesture == "open_hand":
            self.cursor_enabled = True
            print("▶️ Cursor enabled")
    
    def _draw_ultra_fast_ui(self, frame: np.ndarray, result) -> np.ndarray:
        """Draw minimal UI for maximum performance"""
        # Only draw essential information
        h, w = frame.shape[:2]
        
        # FPS in top-left
        fps_color = (0, 255, 0) if self.fps_actual >= 50 else (0, 255, 255) if self.fps_actual >= 30 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.fps_actual:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Sensitivity indicator
        cv2.putText(frame, f"Sensitivity: {int(self.sensitivity * 10)}/9", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Cursor status
        cursor_status = "ON" if self.cursor_enabled else "OFF"
        cursor_color = (0, 255, 0) if self.cursor_enabled else (0, 0, 255)
        cv2.putText(frame, f"Cursor: {cursor_status}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cursor_color, 1)
        
        # Draw minimal hand indicators
        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand_type = handedness[0].category_name
                color = (0, 255, 0) if hand_type == "Right" else (255, 100, 0)
                
                # Just draw a circle at hand center
                if landmarks:
                    # Calculate hand center
                    xs = [lm.x * w for lm in landmarks]
                    ys = [lm.y * h for lm in landmarks]
                    center_x, center_y = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
                    
                    cv2.circle(frame, (center_x, center_y), 15, color, 3)
                    cv2.putText(frame, hand_type[0], (center_x - 10, center_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Instructions (bottom)
        instructions = [
            "RIGHT HAND: Point=Move | Pinch=Click/Drag | Peace=RightClick | 3Fingers=ScrollUp | 4Fingers=ScrollDown",
            "CONTROLS: 1-9=Sensitivity | LEFT HAND: Fist=Disable | Open=Enable | Space=Toggle | ESC=Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 40 + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _draw_minimal_ui(self, frame: np.ndarray) -> np.ndarray:
        """Ultra-minimal UI for skipped frames"""
        # Just show FPS and cursor status
        cv2.putText(frame, f"FPS: {self.fps_actual:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
    
    def run_ultra_fast(self):
        """Main ultra-fast application loop"""
        print("🎥 Starting ULTRA FAST camera system...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera to lower resolution for maximum speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower resolution
        cap.set(cv2.CAP_PROP_FPS, 60)           # Target 60 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Minimize latency
        
        print("✅ Camera optimized: 640x480 @ 60 FPS")
        print("🖐️ Show your RIGHT HAND to control cursor!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process with ultra-fast optimization
                processed_frame = self.process_frame_ultra_fast(frame)
                
                # Display
                cv2.imshow('ULTRA FAST Hand Tracking - Apple Vision Style', processed_frame)
                
                # Handle keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif ord('1') <= key <= ord('9'):  # Sensitivity 1-9
                    sensitivity_level = key - ord('0')
                    self.adjust_sensitivity(sensitivity_level)
                elif key == ord(' '):  # Space to toggle cursor
                    self.cursor_enabled = not self.cursor_enabled
                    print(f"🎯 Cursor {'enabled' if self.cursor_enabled else 'disabled'}")
                elif key == ord('r'):  # Reset
                    self.ema_smoother.reset()
                    self.gesture_states = {k: False for k in self.gesture_states}
                    print("🔄 System reset!")
                    
        except KeyboardInterrupt:
            print("\n⏹️ Stopping ultra-fast hand tracking...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Ultra-fast cleanup complete!")


def main():
    """Main entry point for ultra-fast hand tracking"""
    print("🚀 ULTRA FAST Hand Tracking System")
    print("=" * 50)
    print("🍎 Apple Vision Pro Style Controls")
    print("⚡ Optimized for maximum performance")
    print("🖐️ Right-hand cursor control")
    
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure hand_landmarker.task is in the current directory")
        return
    
    # Create and run ultra-fast tracker
    tracker = UltraFastHandTracker()
    tracker.run_ultra_fast()


if __name__ == "__main__":
    main()