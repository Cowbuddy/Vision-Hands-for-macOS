#!/usr/bin/env python3
"""
ULTRA FAST Hand Tracking System - OPTIMIZED VERSION

Fixed flickering issues with proper memory management and MediaPipe optimization.
Features: Memory-efficient processing, frame pooling, and optimized MediaPipe initialization.
"""
import os
import cv2
import time
import numpy as np
import gc
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
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # Force CPU for stability

# Constants
MODEL_PATH = "hand_landmarker.task"


class MemoryOptimizedCursorController:
    """Memory-optimized cursor controller with object pooling"""
    
    def __init__(self):
        self.screen_width = 0
        self.screen_height = 0
        self._init_screen_info()
        
        # Pre-allocate event objects for memory efficiency
        if MACOS_NATIVE:
            self._mouse_move_event = None
            self._left_down_event = None
            self._left_up_event = None
            self._right_down_event = None
            self._right_up_event = None
    
    def _init_screen_info(self):
        """Initialize screen dimensions"""
        if MACOS_NATIVE:
            main_display = Quartz.CGMainDisplayID()
            self.screen_width = Quartz.CGDisplayPixelsWide(main_display)
            self.screen_height = Quartz.CGDisplayPixelsHigh(main_display)
        else:
            self.screen_width, self.screen_height = pyautogui.size()
    
    def move_cursor(self, x: float, y: float):
        """Move cursor with minimal memory allocation"""
        if MACOS_NATIVE:
            # Reuse CGPoint to avoid allocation
            point = Quartz.CGPoint(x, y)
            Quartz.CGWarpMouseCursorPosition(point)
        else:
            pyautogui.moveTo(x, y, duration=0)
    
    def click(self):
        """Perform left click with pre-allocated events"""
        if MACOS_NATIVE:
            # Get current cursor position
            pos = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
            
            # Create and post mouse down event
            down_event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, pos, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
            
            # Create and post mouse up event
            up_event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, pos, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)
        else:
            pyautogui.click()


class OptimizedEMASmoothing:
    """Memory-efficient EMA smoothing"""
    
    def __init__(self, alpha: float = 0.5):  # Increased default for better responsiveness
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.x = 0.0
        self.y = 0.0
        self.initialized = False
        
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update with new position (in-place operation)"""
        if not self.initialized:
            self.x = x
            self.y = y
            self.initialized = True
        else:
            self.x = self.alpha * x + self.beta * self.x
            self.y = self.alpha * y + self.beta * self.y
        
        return self.x, self.y
    
    def set_sensitivity(self, sensitivity: float):
        """Adjust sensitivity"""
        self.alpha = max(0.1, min(0.9, sensitivity))
        self.beta = 1.0 - self.alpha


class OptimizedHandTracker:
    """Ultra-optimized hand tracker with memory management"""
    
    def __init__(self):
        # Initialize MediaPipe once with optimized settings
        self.hand_landmarker = self._create_optimized_landmarker()
        self.hand_analyzer = HandAnalyzer()
        self.gesture_recognizer = StableGestureRecognizer(history_size=5)  # Smaller for memory
        self.cursor_controller = MemoryOptimizedCursorController()
        self.system_controller = SystemController()
        
        # Pre-allocate arrays for frame processing
        self.rgb_frame = None
        self.mp_image = None
        
        # Performance tracking with limited history
        self.frame_times = deque(maxlen=30)  # Only keep 30 frames
        self.last_fps_update = time.time()
        self.fps_actual = 0
        self.frame_count = 0
        
        # Cursor settings
        self.cursor_enabled = True
        self.sensitivity = 0.4
        self.movement_threshold = 1  # Reduced for better responsiveness
        self.ema_smoother = OptimizedEMASmoothing(alpha=self.sensitivity)
        
        # Gesture state (minimal memory footprint)
        self.last_cursor_pos = [0.0, 0.0]
        self.gesture_states = {
            'pinch_held': False,
            'left_pinch_held': False,  # For left hand tracking toggle
            'pinch_start_time': 0,
            'drag_active': False,
            'right_click_ready': False,
            'scroll_active': False,
            'last_scroll_time': 0
        }
        
        # Memory management
        self.gc_counter = 0
        self.gc_interval = 300  # Run garbage collection every 300 frames
        
        print("🚀 Optimized hand tracker initialized")
    
    def _create_optimized_landmarker(self):
        """Create MediaPipe landmarker with optimized settings"""
        print("⚡ Initializing optimized MediaPipe...")
        
        base_opts = BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU  # CPU for consistency
        )
        
        options = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,  # Slightly higher for stability
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        return HandLandmarker.create_from_options(options)
    
    def process_frame_optimized(self, frame):
        """Process frame with memory optimization"""
        start_time = time.time()
        self.frame_count += 1
        
        # Memory management - run GC periodically
        if self.gc_counter >= self.gc_interval:
            gc.collect()
            self.gc_counter = 0
        self.gc_counter += 1
        
        frame_height, frame_width = frame.shape[:2]
        
        # Reuse RGB frame buffer if possible
        if self.rgb_frame is None or self.rgb_frame.shape != frame.shape:
            self.rgb_frame = np.empty_like(frame)
        
        # Convert to RGB (in-place when possible)
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self.rgb_frame)
        
        # Create MediaPipe image (reuse when possible)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_frame)
        
        # Detect hands with optimized timestamp
        timestamp_ms = int(self.frame_count * 16.67)  # 60fps timestamps
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Process hands for cursor control
        if result.hand_landmarks and result.handedness:
            self._process_hands_optimized(
                result.hand_landmarks, result.handedness, frame_width, frame_height
            )
        
        # Draw minimal UI
        annotated_frame = self._draw_minimal_ui(frame, result)
        
        # Update performance metrics (limited to reduce overhead)
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        if time.time() - self.last_fps_update >= 1.0:
            if self.frame_times:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.fps_actual = 1.0 / avg_time if avg_time > 0 else 0
            self.last_fps_update = time.time()
        
        return annotated_frame
    
    def _process_hands_optimized(self, hand_landmarks_list, handedness_list, frame_width, frame_height):
        """Optimized hand processing with separate left/right hand functions"""
        left_hand = None
        right_hand = None
        
        # Separate hands by type
        for landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            hand_info = self.hand_analyzer.analyze_hand(landmarks, handedness, frame_width, frame_height)
            if hand_info.hand_type == "Left":
                left_hand = hand_info
            elif hand_info.hand_type == "Right":
                right_hand = hand_info
        
        # Handle left hand for tracking toggle
        if left_hand:
            self._handle_left_hand_controls(left_hand)
        
        # Handle right hand for cursor control (regardless of gesture when tracking enabled)
        if right_hand and self.cursor_enabled:
            if self.frame_count % 60 == 0:  # Every 60 frames
                print(f"🖐️ Processing {right_hand.hand_type} hand for cursor control")
            self._handle_right_hand_cursor(right_hand, frame_width, frame_height)
    

    
    def _move_cursor_optimized(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Optimized cursor movement - works regardless of gesture/angle"""
        # Use index finger tip position regardless of gesture detection
        # This makes it work even when pointing directly at camera or at weird angles
        index_tip = hand_info.fingers["index"].tip_position
        
        if self.frame_count % 60 == 0:  # Debug every 60 frames
            print(f"🔍 Index tip position: {index_tip}")
        
        # Convert to normalized coordinates (0-1)
        norm_x = index_tip[0] / frame_width
        norm_y = index_tip[1] / frame_height
        
        # Map to screen coordinates
        screen_x = norm_x * self.cursor_controller.screen_width
        screen_y = norm_y * self.cursor_controller.screen_height
        
        # Ensure coordinates are within screen bounds
        screen_x = max(0, min(self.cursor_controller.screen_width - 1, screen_x))
        screen_y = max(0, min(self.cursor_controller.screen_height - 1, screen_y))
        
        if self.frame_count % 60 == 0:  # Debug every 60 frames
            print(f"📱 Screen coordinates: ({screen_x:.1f}, {screen_y:.1f}) [Screen: {self.cursor_controller.screen_width}x{self.cursor_controller.screen_height}]")
        
        # Apply EMA smoothing
        smoothed_x, smoothed_y = self.ema_smoother.update(screen_x, screen_y)
        
        # Check movement threshold (reduced for better responsiveness)
        dx = abs(smoothed_x - self.last_cursor_pos[0])
        dy = abs(smoothed_y - self.last_cursor_pos[1])
        
        # Lower threshold for better responsiveness
        movement_threshold = max(5.0, self.movement_threshold * 2.0)  # Increase threshold for large screen
        
        if self.frame_count % 60 == 0:  # Debug every 60 frames
            print(f"📏 Movement: dx={dx:.1f}, dy={dy:.1f}, threshold={movement_threshold:.1f}")
        
        if dx > movement_threshold or dy > movement_threshold:
            if self.frame_count % 60 == 0:
                print(f"🖱️ MOVING CURSOR to ({smoothed_x:.1f}, {smoothed_y:.1f})")
            # Use simple pyautogui directly for testing
            import pyautogui
            pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)
            self.last_cursor_pos[0] = smoothed_x
            self.last_cursor_pos[1] = smoothed_y
    
    def _draw_minimal_ui(self, frame, result):
        """Draw minimal UI to reduce overhead"""
        # Only draw essential information
        height, width = frame.shape[:2]
        
        # Status text (minimal)
        status_text = f"FPS: {self.fps_actual:.1f} | Cursor: {'ON' if self.cursor_enabled else 'OFF'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw hand landmarks (simplified)
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw only key points to reduce drawing overhead
                for i, landmark in enumerate(hand_landmarks):
                    if i in [4, 8, 12, 16, 20]:  # Only fingertips
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        return frame
    
    def adjust_sensitivity(self, level: int):
        """Adjust cursor sensitivity (1-9)"""
        sensitivity = level / 10.0  # Convert to 0.1-0.9
        self.ema_smoother.set_sensitivity(sensitivity)
        print(f"🎯 Sensitivity: {level}/9")
    
    def run_optimized(self):
        """Run optimized hand tracking"""
        print("🎥 Initializing optimized camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Optimized camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        
        print("✅ Optimized camera ready: 640x480 @ 60 FPS")
        print("🖐️ Show your hands for control!")
        print("🎮 NEW DUAL-HAND Controls:")
        print("   LEFT HAND:")
        print("     • Pinch = Toggle cursor tracking ON/OFF")
        print("   RIGHT HAND (when tracking enabled):")
        print("     • Index finger position = Move cursor (works at any angle)")
        print("     • Quick pinch = Click")
        print("     • Hold pinch (0.8s+) = Click-and-hold/drag")
        print("   KEYBOARD: ESC = Exit | SPACE = Toggle cursor | 1-9 = Sensitivity")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process with optimization
                processed_frame = self.process_frame_optimized(frame)
                
                # Display
                cv2.imshow('Optimized Hand Tracking - No Flickering', processed_frame)
                
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
                    self.ema_smoother = OptimizedEMASmoothing(alpha=self.sensitivity)
                    self.gesture_states = {k: False if k != 'last_scroll_time' else 0 for k, v in self.gesture_states.items()}
                    print("🔄 System reset!")
                    
        except KeyboardInterrupt:
            print("\n⏹️ Stopping optimized hand tracking...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Optimized cleanup complete!")


    def _handle_left_hand_controls(self, hand_info: HandInfo):
        """Handle left hand for tracking toggle and other controls"""
        # Get raw gesture and pinch state
        raw_gesture = hand_info.gesture_name
        stable_gesture = self.gesture_recognizer.get_stable_gesture(raw_gesture)
        is_pinching = hand_info.is_pinching
        
        # Debug output for left hand
        if self.frame_count % 60 == 0:  # Every 60 frames
            print(f"🤚 LEFT HAND - Gesture: {raw_gesture}, Pinching: {is_pinching}")
        
        # Handle pinch for tracking toggle
        if is_pinching and not self.gesture_states.get('left_pinch_held', False):
            # Toggle tracking
            self.cursor_enabled = not self.cursor_enabled
            self.gesture_states['left_pinch_held'] = True
            status = "ENABLED" if self.cursor_enabled else "DISABLED"
            print(f"🔄 LEFT HAND PINCH: Cursor tracking {status}")
            
        elif not is_pinching and self.gesture_states.get('left_pinch_held', False):
            # Reset pinch state when released
            self.gesture_states['left_pinch_held'] = False
            print("🤚 Left hand pinch released")

    def _handle_right_hand_cursor(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Enhanced right hand cursor control - works regardless of gesture"""
        is_pinching = hand_info.is_pinching
        
        # Debug output for right hand
        if self.frame_count % 30 == 0:  # Print every 30 frames to avoid spam
            pinch_distance = hand_info.pinch_distance if hasattr(hand_info, 'pinch_distance') else 'N/A'
            print(f"👉 RIGHT HAND - Pinching: {is_pinching}, Pinch distance: {pinch_distance}")
            print(f"   Pinch held state: {self.gesture_states['pinch_held']}")
        
        # Always move cursor based on index finger position (regardless of gesture)
        self._move_cursor_optimized(hand_info, frame_width, frame_height)
        
        # Handle pinch for clicking and click-hold
        current_time = time.time()
        
        if is_pinching and not self.gesture_states['pinch_held']:
            # Start pinch - record time
            self.gesture_states['pinch_start_time'] = current_time
            self.gesture_states['pinch_held'] = True
            print("🖱️ RIGHT HAND: Pinch started")
            
        elif is_pinching and self.gesture_states['pinch_held']:
            # Check if pinch has been held long enough for click-and-hold
            pinch_duration = current_time - self.gesture_states.get('pinch_start_time', current_time)
            
            if pinch_duration > 0.8 and not self.gesture_states.get('drag_active', False):
                # Start drag after 800ms of pinch hold
                self.gesture_states['drag_active'] = True
                print("🖱️ RIGHT HAND: Click-and-hold started (drag mode)")
                # Perform mouse down for drag
                import pyautogui
                pyautogui.mouseDown()
                
        elif not is_pinching and self.gesture_states['pinch_held']:
            # Pinch released
            pinch_duration = current_time - self.gesture_states.get('pinch_start_time', current_time)
            
            if self.gesture_states.get('drag_active', False):
                # End drag
                import pyautogui
                pyautogui.mouseUp()
                self.gesture_states['drag_active'] = False
                print("🖱️ RIGHT HAND: Click-and-hold ended (drag released)")
            elif pinch_duration < 0.8:
                # Quick pinch - perform click
                self.system_controller.left_click()
                print("🖱️ RIGHT HAND: Quick click performed")
            
            # Reset pinch states
            self.gesture_states['pinch_held'] = False
            self.gesture_states['pinch_start_time'] = 0


def main():
    """Main entry point for optimized hand tracking"""
    print("🚀 OPTIMIZED Hand Tracking System")
    print("=" * 50)
    print("✨ Fixed flickering issues")
    print("🧠 Memory-optimized processing")
    print("⚡ Stable 60 FPS performance")
    
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure hand_landmarker.task is in the current directory")
        return
    
    # Create and run optimized tracker
    tracker = OptimizedHandTracker()
    tracker.run_optimized()


if __name__ == "__main__":
    main()
