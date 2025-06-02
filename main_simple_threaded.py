#!/usr/bin/env python3
"""
WORKING THREADED Hand Tracking System

Simple threaded implementation without complex dependencies.
Features basic threading for frame processing optimization.
"""
import os
import cv2
import time
import threading
import queue
import numpy as np
from typing import Optional, Tuple
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

# Import components
from src.models import HandInfo, HandLandmarks
from src.hand_analyzer import HandAnalyzer
from src.enhanced_gesture_recognition import StableGestureRecognizer
from src.system_controller import SystemController

# macOS native cursor control
try:
    import Quartz
    MACOS_NATIVE = True
    print("✅ macOS native cursor APIs loaded")
except ImportError:
    import pyautogui
    MACOS_NATIVE = False
    print("⚠️  Using pyautogui (install pyobjc for native speed)")

# Configure for performance
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Constants
MODEL_PATH = "hand_landmarker.task"


class SimpleCursorController:
    """Simple cursor controller for threaded operation"""
    
    def __init__(self):
        if MACOS_NATIVE:
            main_display = Quartz.CGMainDisplayID()
            self.screen_width = Quartz.CGDisplayPixelsWide(main_display)
            self.screen_height = Quartz.CGDisplayPixelsHigh(main_display)
        else:
            self.screen_width, self.screen_height = pyautogui.size()
    
    def move_cursor(self, x: float, y: float):
        """Move cursor efficiently"""
        if MACOS_NATIVE:
            point = Quartz.CGPoint(x, y)
            Quartz.CGWarpMouseCursorPosition(point)
        else:
            pyautogui.moveTo(x, y, duration=0)
    
    def click(self):
        """Perform click"""
        if MACOS_NATIVE:
            pos = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
            down_event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, pos, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
            up_event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, pos, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)
        else:
            pyautogui.click()


class SimpleThreadedTracker:
    """Simple threaded hand tracker"""
    
    def __init__(self):
        # Core components
        self.hand_landmarker = self._create_landmarker()
        self.hand_analyzer = HandAnalyzer()
        self.gesture_recognizer = StableGestureRecognizer(history_size=5)
        self.cursor_controller = SimpleCursorController()
        self.system_controller = SystemController()
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.processing_thread = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_actual = 0
        
        # Cursor control
        self.cursor_enabled = True
        self.sensitivity = 0.4
        self.last_cursor_pos = [0.0, 0.0]
        
        # Gesture states
        self.gesture_states = {
            'pinch_held': False,
            'last_scroll_time': 0
        }
        
        print("🧵 Simple threaded tracker initialized")
    
    def _create_landmarker(self):
        """Create MediaPipe landmarker"""
        base_opts = BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU
        )
        
        options = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        return HandLandmarker.create_from_options(options)
    
    def _processing_worker(self):
        """Worker thread for processing frames"""
        print("🧵 Processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                frame, timestamp = frame_data
                height, width = frame.shape[:2]
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect hands
                result = self.hand_landmarker.detect_for_video(mp_image, timestamp)
                
                # Process hands
                processed_result = self._process_hands_simple(result, width, height)
                
                # Put result in output queue
                if not self.result_queue.full():
                    self.result_queue.put((frame, processed_result))
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Processing error: {e}")
                break
        
        print("🧵 Processing thread stopped")
    
    def _process_hands_simple(self, result, width, height):
        """Simple hand processing for threading"""
        if not result.hand_landmarks or not result.handedness:
            return None
        
        # Find right hand
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            if handedness[0].display_name == "Right":
                hand_info = self.hand_analyzer.analyze_hand(landmarks, handedness, width, height)
                
                if self.cursor_enabled:
                    self._handle_cursor_threaded(hand_info, width, height)
                
                return hand_info
        
        return None
    
    def _handle_cursor_threaded(self, hand_info: HandInfo, width: int, height: int):
        """Handle cursor in threaded mode"""
        gesture = self.gesture_recognizer.get_stable_gesture(hand_info)
        
        if gesture == "point":
            # Move cursor
            if hand_info.fingers["index"].is_extended:
                index_tip = hand_info.fingers["index"].tip_position
                screen_x = (index_tip[0] / width) * self.cursor_controller.screen_width
                screen_y = (index_tip[1] / height) * self.cursor_controller.screen_height
                
                # Simple movement threshold
                dx = abs(screen_x - self.last_cursor_pos[0])
                dy = abs(screen_y - self.last_cursor_pos[1])
                
                if dx > 3 or dy > 3:
                    self.cursor_controller.move_cursor(screen_x, screen_y)
                    self.last_cursor_pos[0] = screen_x
                    self.last_cursor_pos[1] = screen_y
        
        elif gesture == "pinch":
            # Click
            if not self.gesture_states['pinch_held']:
                self.cursor_controller.click()
                self.gesture_states['pinch_held'] = True
                print("🖱️ Click")
        
        elif gesture in ["three", "four"]:
            # Scrolling
            current_time = time.time()
            if current_time - self.gesture_states['last_scroll_time'] > 0.3:
                if gesture == "three":
                    self.system_controller.scroll(0, 3)
                    print("⬆️ Scroll up")
                elif gesture == "four":
                    self.system_controller.scroll(0, -3)
                    print("⬇️ Scroll down")
                self.gesture_states['last_scroll_time'] = current_time
        
        else:
            # Reset states
            self.gesture_states['pinch_held'] = False
    
    def _draw_simple_ui(self, frame, hand_info):
        """Draw minimal UI"""
        height, width = frame.shape[:2]
        
        # FPS counter
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps_actual = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        # Status
        status = f"Threaded FPS: {self.fps_actual} | Cursor: {'ON' if self.cursor_enabled else 'OFF'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Queue status
        queue_status = f"Queue: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}"
        cv2.putText(frame, queue_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Run threaded hand tracking"""
        print("🎥 Starting threaded camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Optimize camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✅ Threaded camera ready")
        print("🧵 Starting processing thread...")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("🎮 Controls:")
        print("   Point = Move cursor | Pinch = Click")
        print("   3 fingers = Scroll up | 4 fingers = Scroll down")
        print("   SPACE = Toggle cursor | ESC = Exit")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Put frame in processing queue (non-blocking)
                timestamp_ms = int(frame_count * 16.67)
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), timestamp_ms))
                
                # Try to get processed result
                processed_hand = None
                try:
                    display_frame, processed_hand = self.result_queue.get_nowait()
                    frame = display_frame  # Use processed frame if available
                except queue.Empty:
                    pass  # Use current frame
                
                # Draw UI
                annotated_frame = self._draw_simple_ui(frame, processed_hand)
                
                # Display
                cv2.imshow('Simple Threaded Hand Tracking', annotated_frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # Space
                    self.cursor_enabled = not self.cursor_enabled
                    print(f"🎯 Cursor {'enabled' if self.cursor_enabled else 'disabled'}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping threaded tracker...")
        
        finally:
            # Stop processing thread
            self.stop_event.set()
            if self.processing_thread:
                self.processing_thread.join(timeout=2)
            
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Threaded cleanup complete!")


def main():
    """Main entry point"""
    print("🚀 SIMPLE THREADED Hand Tracking System")
    print("=" * 50)
    print("🧵 Multi-threaded frame processing")
    print("⚡ Optimized for performance")
    
    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    # Create and run tracker
    tracker = SimpleThreadedTracker()
    tracker.run()


if __name__ == "__main__":
    main()
