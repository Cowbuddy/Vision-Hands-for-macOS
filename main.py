#!/usr/bin/env python3
"""
Advanced Hand Tracking System with Gesture Recognition and Cursor Control

A comprehensive computer vision system for real-time hand tracking, gesture recognition,
and macOS system control using MediaPipe and OpenCV. Features modular architecture,
left/right hand separation, and intelligent cursor control.
"""
import os
import cv2
import time
import numpy as np
from typing import Optional

# Import MediaPipe
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode
)

# Import our modular components
from src.models import HandInfo, HandLandmarks
from src.hand_analyzer import HandAnalyzer
from src.gesture_recognition import GestureRecognizer
from src.cursor_controller import CursorController
from src.system_controller import SystemController

# Configure environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Constants
MODEL_PATH = "hand_landmarker.task"


def create_hand_landmarker():
    """Create optimized hand landmarker for M4 Pro"""
    print("🔧 Initializing modular hand tracking system...")
    print("🚀 Using optimized CPU configuration for M4 Pro...")
    
    base_opts = BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=BaseOptions.Delegate.GPU
    )
    
    options = HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    return HandLandmarker.create_from_options(options)


class HandTracker:
    """Advanced hand tracking application with gesture recognition and cursor control"""
    
    def __init__(self):
        # Initialize MediaPipe components
        self.hand_landmarker = create_hand_landmarker()
        
        # Initialize our modular components
        self.hand_analyzer = HandAnalyzer()
        self.gesture_recognizer = GestureRecognizer()
        self.system_controller = SystemController()
        self.cursor_controller = CursorController(self.system_controller)
        
        # State tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        print("✅ Modular hand tracking system initialized!")
        print("📝 Features:")
        print("   • Gesture recognition with transitions")
        print("   • Cursor control with pinch gestures")
        print("   • Mission Control (fist → open hand)")
        print("   • Left/right hand classification")
        print("   • Enhanced visual feedback")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with full modular pipeline"""
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # Convert BGR to RGBA for MediaPipe
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        
        # Detect hands
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Process each detected hand
        hands_info = []
        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                # Analyze hand using modular components
                hand_info = self.hand_analyzer.analyze_hand(
                    landmarks, handedness, frame_width, frame_height
                )
                
                # Recognize gestures
                hand_info.gesture_name = self.gesture_recognizer.recognize_gesture(
                    hand_info.fingers, hand_info.is_fist, hand_info.is_pinching
                )
                
                # Detect gesture transitions
                transition = self.gesture_recognizer.detect_transition(hand_info.gesture_name)
                
                # Handle Mission Control gesture (fist → open hand)
                if transition and transition.from_gesture == "fist" and transition.to_gesture == "open_hand":
                    print("🚀 Mission Control gesture detected!")
                    self.system_controller.trigger_mission_control()
                
                hands_info.append(hand_info)
        
        # Handle cursor control - now uses left hand by default to avoid interference
        self.cursor_controller.handle_multi_hand_cursor_control(hands_info, frame_width, frame_height)
        
        # Draw visualization
        annotated_frame = self.draw_annotations(frame, hands_info)
        
        # Add system info
        annotated_frame = self.draw_system_info(annotated_frame)
        
        return annotated_frame
    
    def draw_annotations(self, frame: np.ndarray, hands_info: list) -> np.ndarray:
        """Draw all hand annotations and visual feedback"""
        annotated_frame = frame.copy()
        
        for hand_info in hands_info:
            # Choose color based on hand type
            color = (0, 255, 0) if hand_info.hand_type == "Right" else (255, 0, 0)
            
            # Draw landmarks
            for x, y in hand_info.landmarks:
                cv2.circle(annotated_frame, (int(x), int(y)), 3, color, -1)
            
            # Draw hand connections
            self._draw_hand_connections(annotated_frame, hand_info.landmarks, color)
            
            # Draw pinch visualization
            if hand_info.is_pinching:
                thumb_tip = hand_info.landmarks[HandLandmarks.THUMB_TIP]
                index_tip = hand_info.landmarks[HandLandmarks.INDEX_TIP]
                cv2.line(annotated_frame, 
                        (int(thumb_tip[0]), int(thumb_tip[1])),
                        (int(index_tip[0]), int(index_tip[1])),
                        (0, 255, 255), 3)
            
            # Draw hand info
            center_x, center_y = map(int, hand_info.center_position)
            
            # Hand type and gesture
            info_text = f"{hand_info.hand_type} - {hand_info.gesture_name}"
            cv2.putText(annotated_frame, info_text, (center_x - 80, center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Finger states
            finger_states = [f"{name[:3]}:{'✓' if finger.is_extended else '✗'}" 
                           for name, finger in hand_info.fingers.items()]
            finger_text = " ".join(finger_states)
            cv2.putText(annotated_frame, finger_text, (center_x - 120, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Pinch info
            if hand_info.is_pinching:
                pinch_text = f"Pinch: {hand_info.pinch_distance:.1f}px"
                cv2.putText(annotated_frame, pinch_text, (center_x - 60, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return annotated_frame
    
    def _draw_hand_connections(self, frame: np.ndarray, landmarks: list, color: tuple):
        """Draw hand skeleton connections"""
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                cv2.line(frame, start_point, end_point, color, 2)
    
    def draw_system_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw system status and cursor control info"""
        h, w = frame.shape[:2]
        
        # FPS calculation
        current_time = time.time()
        self.fps_counter += 1
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
            self.current_fps = fps
        
        # Status info
        fps_text = f"FPS: {getattr(self, 'current_fps', 0):.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cursor_status = self.cursor_controller.get_status()
        cursor_color = (0, 255, 0) if self.cursor_controller.cursor_enabled else (0, 0, 255)
        cv2.putText(frame, f"Cursor: {cursor_status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cursor_color, 2)
        
        # Instructions
        instructions = [
            "CONTROLS:",
            "• LEFT HAND: Pinch → Toggle cursor (default)",
            "• LEFT HAND: Double pinch → Left click",
            "• ANY HAND: Fist → Open hand → Mission Control",
            "• L key → Switch to LEFT hand cursor",
            "• R key → Switch to RIGHT hand cursor", 
            "• Q to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 120 + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("🎥 Starting camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized. Starting hand tracking...")
        print("📱 Show your hands to the camera!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame through modular pipeline
                processed_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow('Modular Hand Tracking', processed_frame)
                
                # Handle exit and controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):  # Switch to left hand
                    self.cursor_controller.set_preferred_hand("Left")
                elif key == ord('r'):  # Switch to right hand
                    self.cursor_controller.set_preferred_hand("Right")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping hand tracking...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup complete!")


def main():
    """Main entry point"""
    print("🚀 Starting Modular Hand Tracking System")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the hand_landmarker.task file is in the current directory")
        return
    
    # Create and run the tracker
    tracker = HandTracker()
    tracker.run()


if __name__ == "__main__":
    main()
