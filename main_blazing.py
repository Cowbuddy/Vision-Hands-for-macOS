#!/usr/bin/env python3
"""
BLAZING FAST Hand Tracking System v3.0 with Vim-like Controls

A high-performance computer vision system optimized for 60+ FPS with vim-like modal controls.
Right-hand dominant design with intelligent gesture recognition and zero-interference operation.
"""
import os
import cv2
import time
import numpy as np
import threading
from typing import Optional, List, Dict, Any

# Import MediaPipe
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode
)

# Import our enhanced modular components
from src.models import HandInfo, HandLandmarks
from src.hand_analyzer import HandAnalyzer
from src.enhanced_gesture_recognition import StableGestureRecognizer, GestureFilter
from src.vim_cursor_controller import VimCursorController
from src.system_controller import SystemController
from src.performance_optimizer import FrameOptimizer, VisualizationOptimizer, FPSMonitor

# Configure environment for max performance
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Constants
MODEL_PATH = "hand_landmarker.task"


def create_optimized_hand_landmarker():
    """Create highly optimized hand landmarker for blazing performance"""
    print("🚀 Initializing BLAZING FAST hand tracking system...")
    print("⚡ M4 Pro optimization: MAXIMUM PERFORMANCE mode")
    
    base_opts = BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=BaseOptions.Delegate.CPU  # CPU is actually faster on M4 Pro
    )
    
    options = HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,  # Slightly lower for speed
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    return HandLandmarker.create_from_options(options)


class BlazingHandTracker:
    """Blazing fast hand tracking with vim-like controls and zero interference"""
    
    def __init__(self):
        # Initialize MediaPipe with optimization
        self.hand_landmarker = create_optimized_hand_landmarker()
        
        # Initialize enhanced modular components
        self.hand_analyzer = HandAnalyzer()
        self.gesture_recognizer = StableGestureRecognizer(history_size=12)
        self.gesture_filter = GestureFilter()
        self.system_controller = SystemController()
        self.cursor_controller = VimCursorController(self.system_controller)
        
        # Performance optimization components
        self.frame_optimizer = FrameOptimizer()
        self.vis_optimizer = VisualizationOptimizer()
        self.fps_monitor = FPSMonitor(target_fps=60)
        
        # State tracking
        self.frame_count = 0
        self.last_hands_info = []
        self.performance_mode = "BALANCED"  # SPEED, BALANCED, QUALITY
        
        # Threading for non-critical operations
        self.ui_thread = None
        self.ui_queue = []
        
        print("✅ BLAZING FAST hand tracking system initialized!")
        print("🎮 VIM-LIKE CONTROLS:")
        print("   RIGHT HAND (Cursor):")
        print("     • Point finger → Move cursor")
        print("     • Pinch → Click")
        print("     • Open hand → Free movement")
        print("     • Fist → Hold position")
        print("   LEFT HAND (Commands):")
        print("     • Point → Enable cursor mode")
        print("     • Peace → Precision mode")
        print("     • Fist → Normal mode")
        print("     • Thumb up → Mission Control")
        print("     • Call me → Special action")
    
    def process_frame_blazing_fast(self, frame: np.ndarray) -> np.ndarray:
        """Ultra-optimized frame processing for maximum FPS"""
        start_time = time.time()
        self.frame_count += 1
        
        # Performance check - skip processing if needed
        if not self.frame_optimizer.should_process_frame():
            return self._draw_minimal_ui(frame)
        
        frame_height, frame_width = frame.shape[:2]
        
        # Optimize frame for processing
        optimized_frame, _ = self.frame_optimizer.optimize_frame(frame)
        
        # Convert for MediaPipe (RGB is faster than RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=optimized_frame)
        
        # Detect hands with optimized timestamp
        timestamp_ms = int(self.frame_count * 16.67)  # Assume ~60fps for consistent timestamps
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Process detected hands
        hands_info = []
        if result.hand_landmarks and result.handedness:
            hands_info = self._process_hands_optimized(
                result.hand_landmarks, result.handedness, frame_width, frame_height
            )
        
        # Handle vim-like control with separated hands
        command_result = self._handle_vim_controls(hands_info, frame_width, frame_height)
        
        # Execute special commands
        if command_result:
            self._execute_command(command_result)
        
        # Draw optimized visualization
        annotated_frame = self._draw_blazing_ui(frame, hands_info)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        fps = self.fps_monitor.update()
        
        # Dynamic performance adjustment
        self.frame_optimizer.adjust_performance(fps)
        
        return annotated_frame
    
    def _process_hands_optimized(self, hand_landmarks_list, handedness_list, frame_width, frame_height) -> List[HandInfo]:
        """Optimized hand processing with stable gesture recognition"""
        hands_info = []
        
        for landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            # Basic hand analysis
            hand_info = self.hand_analyzer.analyze_hand(
                landmarks, handedness, frame_width, frame_height
            )
            
            # Stable gesture recognition
            raw_gesture = self.gesture_recognizer.recognize_gesture(
                hand_info.fingers, hand_info.is_fist, hand_info.is_pinching
            )
            
            # Get stable gesture with confidence
            stable_gesture = self.gesture_recognizer.get_stable_gesture(raw_gesture)
            confidence = self.gesture_recognizer.get_gesture_confidence(raw_gesture)
            
            # Filter gesture to reduce noise
            filtered_gesture = self.gesture_filter.filter_gesture(stable_gesture or raw_gesture, confidence)
            
            # Update hand info with stable gesture
            hand_info.gesture_name = filtered_gesture or "unknown"
            
            hands_info.append(hand_info)
        
        # Reset stability if no hands detected
        if not hands_info:
            self.gesture_recognizer.reset_stability()
        
        self.last_hands_info = hands_info
        return hands_info
    
    def _handle_vim_controls(self, hands_info: List[HandInfo], frame_width: int, frame_height: int) -> Optional[str]:
        """Handle vim-like dual-hand controls with zero interference"""
        if not hands_info:
            return None
        
        # Use vim cursor controller for clean hand separation
        return self.cursor_controller.handle_dual_hand_control(hands_info, frame_width, frame_height)
    
    def _execute_command(self, command: str):
        """Execute special commands from left hand gestures"""
        if command == "QUICK_ACTION":
            # Mission Control with stable gesture detection
            print("🚀 Mission Control activated via stable gesture!")
            self.system_controller.trigger_mission_control()
        elif command == "SPECIAL_ACTION":
            print("✨ Special action triggered!")
            # Could trigger other system actions
        elif "MODE" in command:
            print(f"🎯 Mode changed: {command}")
    
    def _draw_blazing_ui(self, frame: np.ndarray, hands_info: List[HandInfo]) -> np.ndarray:
        """Draw optimized UI for maximum performance"""
        if self.performance_mode == "SPEED":
            return self._draw_minimal_ui(frame, hands_info)
        
        annotated_frame = frame.copy()
        
        # Draw hands with optimized visualization
        for hand_info in hands_info:
            color = (0, 255, 0) if hand_info.hand_type == "Right" else (255, 100, 0)
            
            # Minimal landmark drawing for performance
            if self.vis_optimizer.draw_landmarks:
                self.vis_optimizer.draw_minimal_hand(annotated_frame, hand_info, color)
            
            # Gesture indicators
            self.vis_optimizer.draw_gesture_indicator(annotated_frame, hand_info, color)
            
            # Special indicators for vim modes
            self._draw_vim_indicators(annotated_frame, hand_info, color)
        
        # System status
        self._draw_performance_info(annotated_frame)
        
        return annotated_frame
    
    def _draw_minimal_ui(self, frame: np.ndarray, hands_info: List[HandInfo] = None) -> np.ndarray:
        """Ultra-minimal UI for maximum speed"""
        if hands_info:
            for hand_info in hands_info:
                center_x, center_y = map(int, hand_info.center_position)
                color = (0, 255, 0) if hand_info.hand_type == "Right" else (255, 100, 0)
                
                # Just draw a circle at hand center
                cv2.circle(frame, (center_x, center_y), 8, color, -1)
                
                # Minimal gesture text
                cv2.putText(frame, f"{hand_info.hand_type[0]}:{hand_info.gesture_name[:4]}", 
                           (center_x - 20, center_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Just FPS
        fps = self.fps_monitor.get_average_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _draw_vim_indicators(self, frame: np.ndarray, hand_info: HandInfo, color):
        """Draw vim-specific mode indicators"""
        center_x, center_y = map(int, hand_info.center_position)
        
        if hand_info.hand_type == "Right":
            # Right hand - cursor indicators
            cursor_status = self.cursor_controller.get_status()
            mode_text = f"CURSOR: {cursor_status['mode']}"
            cv2.putText(frame, mode_text, (center_x - 40, center_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if cursor_status['precision_mode']:
                cv2.putText(frame, "PRECISION", (center_x - 30, center_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        elif hand_info.hand_type == "Left":
            # Left hand - command indicators
            mode_info = self.gesture_recognizer.get_mode_info()
            command_text = f"CMD: {mode_info['mode']}"
            cv2.putText(frame, command_text, (center_x - 40, center_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_performance_info(self, frame: np.ndarray):
        """Draw performance and control information"""
        h, w = frame.shape[:2]
        
        # Performance info
        fps = self.fps_monitor.get_average_fps()
        perf_status = self.fps_monitor.get_performance_status()
        
        fps_color = (0, 255, 0) if fps >= 50 else (0, 255, 255) if fps >= 30 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f} ({perf_status})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Cursor status
        cursor_status = self.cursor_controller.get_status()
        cursor_color = (0, 255, 0) if cursor_status['cursor_enabled'] else (128, 128, 128)
        cv2.putText(frame, f"Mode: {cursor_status['mode']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cursor_color, 2)
        
        # Controls (compact)
        controls = [
            "RIGHT: Point=Move, Pinch=Click, Open=Free, Fist=Hold",
            "LEFT: Point=Cursor, Peace=Precision, Fist=Normal, Thumb=Mission",
            "Keys: S=Speed, B=Balanced, Q=Quality, V=Minimal, ESC=Quit"
        ]
        
        for i, control in enumerate(controls):
            y_pos = h - 60 + (i * 20)
            cv2.putText(frame, control, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_blazing_fast(self):
        """Main application loop optimized for maximum performance"""
        print("🎥 Starting BLAZING FAST camera system...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Optimize camera settings for maximum FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
        
        print("✅ Camera optimized for 60+ FPS")
        print("🖐️ Show your hands for blazing fast tracking!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip frame horizontally for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process frame with blazing optimization
                processed_frame = self.process_frame_blazing_fast(frame)
                
                # Display result
                cv2.imshow('BLAZING Hand Tracking v3.0', processed_frame)
                
                # Handle keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # Speed mode
                    self.performance_mode = "SPEED"
                    print("⚡ SPEED mode activated!")
                elif key == ord('b'):  # Balanced mode
                    self.performance_mode = "BALANCED"
                    print("⚖️ BALANCED mode activated!")
                elif key == ord('q'):  # Quality mode
                    self.performance_mode = "QUALITY"
                    print("💎 QUALITY mode activated!")
                elif key == ord('v'):  # Toggle visualization
                    self.vis_optimizer.toggle_detail_level()
                elif key == ord('r'):  # Reset
                    self.cursor_controller.reset_state()
                    print("🔄 System reset!")
                    
        except KeyboardInterrupt:
            print("\n⏹️ Stopping blazing hand tracking...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Blazing cleanup complete!")


def main():
    """Main entry point for blazing fast hand tracking"""
    print("🚀 Starting BLAZING FAST Hand Tracking System v3.0")
    print("=" * 60)
    print("⚡ Optimized for 60+ FPS on Apple M4 Pro")
    print("🎮 Vim-like controls with zero interference")
    print("🖐️ Right-hand dominant design")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the hand_landmarker.task file is in the current directory")
        return
    
    # Create and run the blazing tracker
    tracker = BlazingHandTracker()
    tracker.run_blazing_fast()


if __name__ == "__main__":
    main()
