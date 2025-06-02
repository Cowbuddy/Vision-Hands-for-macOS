"""
Threading-optimized hand tracker for maximum performance
"""
import cv2
import time
import threading
import queue
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque

from .base_hand_tracker import BaseHandTracker
from .models import HandInfo
from .enhanced_gesture_recognition import StableGestureRecognizer
from .vim_cursor_controller import VimCursorController
from .performance_optimizer import FrameOptimizer, FPSMonitor


class ThreadedHandTracker(BaseHandTracker):
    """Hand tracker with threading optimization for maximum performance"""
    
    def __init__(self, model_path: str = "hand_landmarker.task"):
        super().__init__(model_path)
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=3)  # Small queue to prevent lag
        self.result_queue = queue.Queue(maxsize=3)
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Performance optimization
        self.frame_optimizer = FrameOptimizer()
        self.fps_monitor = FPSMonitor(target_fps=60)
        
        # Enhanced components
        self.gesture_recognizer = StableGestureRecognizer(history_size=8)
        self.cursor_controller = VimCursorController(self.system_controller)
        
        # Frame skipping for performance
        self.frame_skip_count = 0
        self.skip_frames = 1  # Process every frame by default
        
        print("✅ THREADED hand tracking system initialized!")
        print("🧵 Multi-threaded processing for maximum performance")
    
    def create_hand_landmarker(self):
        """Create optimized hand landmarker for threading"""
        options = self.create_base_landmarker_options(confidence=0.6, cpu_delegate=True)
        from mediapipe.tasks.python.vision import HandLandmarker
        return HandLandmarker.create_from_options(options)
    
    def get_camera_config(self) -> Dict[str, int]:
        """Get camera configuration optimized for threading"""
        return {
            'width': 640,   # Lower resolution for speed
            'height': 480,
            'fps': 60
        }
    
    def get_window_title(self) -> str:
        return "THREADED Hand Tracking - Ultra Performance"
    
    def print_startup_info(self):
        print("🚀 Starting THREADED Hand Tracking System")
        print("=" * 50)
        print("🧵 Multi-threaded processing enabled")
        print("⚡ Optimized for maximum performance")
        print("🎮 Vim-like controls")
    
    def print_instructions(self):
        print("🖐️ Show your hands for ultra-fast threaded tracking!")
        print("📝 Controls:")
        print("   • RIGHT HAND: Point=Move, Pinch=Click, Peace=RightClick")
        print("   • LEFT HAND: Point=Cursor mode, Peace=Precision, Fist=Normal")
        print("   • ESC=Quit, R=Reset")
    
    def processing_worker(self):
        """Worker thread for processing frames"""
        hand_landmarker = self.create_hand_landmarker()
        
        while not self.stop_event.is_set():
            try:
                # Get frame from queue with timeout
                frame, timestamp = self.frame_queue.get(timeout=0.1)
                
                # Process frame
                result = self._process_frame_internal(frame, hand_landmarker, timestamp)
                
                # Put result in output queue
                try:
                    self.result_queue.put_nowait((result, timestamp))
                except queue.Full:
                    # Drop old results if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait((result, timestamp))
                    except queue.Empty:
                        pass
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                continue
    
    def _process_frame_internal(self, frame: np.ndarray, hand_landmarker, timestamp: int) -> Tuple[np.ndarray, Optional[Any]]:
        """Internal frame processing for worker thread"""
        frame_height, frame_width = frame.shape[:2]
        
        # Optimize frame for processing
        optimized_frame, _ = self.frame_optimizer.optimize_frame(frame)
        
        # Convert for MediaPipe
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=optimized_frame)
        
        # Detect hands
        result = hand_landmarker.detect_for_video(mp_image, timestamp)
        
        # Process hands if detected
        hands_info = []
        if result.hand_landmarks and result.handedness:
            hands_info = self._process_detected_hands(
                result.hand_landmarks, result.handedness, frame_width, frame_height
            )
        
        return frame.copy(), hands_info
    
    def _process_detected_hands(self, hand_landmarks_list, handedness_list, frame_width, frame_height):
        """Process detected hands with gesture recognition"""
        hands_info = []
        
        for landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            # Analyze hand
            hand_info = self.hand_analyzer.analyze_hand(
                landmarks, handedness, frame_width, frame_height
            )
            
            # Recognize gesture
            gesture = self.gesture_recognizer.recognize_gesture(
                hand_info.fingers, hand_info.is_fist, hand_info.is_pinching
            )
            hand_info.gesture_name = gesture
            
            hands_info.append(hand_info)
        
        return hands_info
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using threading"""
        current_time = int(time.time() * 1000)
        
        # Skip frames if needed for performance
        self.frame_skip_count += 1
        if self.frame_skip_count % self.skip_frames != 0:
            return self._draw_minimal_ui(frame)
        
        # Add frame to processing queue
        try:
            self.frame_queue.put_nowait((frame.copy(), current_time))
        except queue.Full:
            # Drop old frames if queue is full
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame.copy(), current_time))
            except queue.Empty:
                pass
        
        # Get processed result if available
        processed_frame = frame.copy()
        hands_info = []
        
        try:
            processed_frame, hands_info = self.result_queue.get_nowait()
        except queue.Empty:
            # No result ready, use previous frame
            pass
        
        # Handle controls if we have hand data
        if hands_info:
            self._handle_threaded_controls(hands_info, frame.shape[1], frame.shape[0])
        
        # Draw UI
        return self._draw_threaded_ui(processed_frame, hands_info)
    
    def _handle_threaded_controls(self, hands_info, frame_width, frame_height):
        """Handle controls in main thread"""
        # Use vim cursor controller for hand separation
        self.cursor_controller.handle_dual_hand_control(hands_info, frame_width, frame_height)
    
    def _draw_threaded_ui(self, frame: np.ndarray, hands_info) -> np.ndarray:
        """Draw UI for threaded version"""
        annotated_frame = frame.copy()
        
        # Draw hands
        for hand_info in hands_info:
            color = (0, 255, 0) if hand_info.hand_type == "Right" else (255, 100, 0)
            center_x, center_y = map(int, hand_info.center_position)
            
            # Draw hand center
            cv2.circle(annotated_frame, (center_x, center_y), 8, color, -1)
            
            # Draw gesture info
            cv2.putText(annotated_frame, f"{hand_info.hand_type}: {hand_info.gesture_name}", 
                       (center_x - 60, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Performance info
        fps = self.fps_monitor.get_average_fps()
        queue_size = self.frame_queue.qsize()
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Queue: {queue_size}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Threading status
        thread_status = "ACTIVE" if self.processing_thread and self.processing_thread.is_alive() else "INACTIVE"
        cv2.putText(annotated_frame, f"Threading: {thread_status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        h = annotated_frame.shape[0]
        instructions = [
            "THREADED PROCESSING: RIGHT=Cursor, LEFT=Commands",
            "ESC=Quit | R=Reset | Threading enabled for max performance"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 40 + (i * 20)
            cv2.putText(annotated_frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_frame
    
    def _draw_minimal_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw minimal UI for skipped frames"""
        fps = self.fps_monitor.get_average_fps()
        cv2.putText(frame, f"FPS: {fps:.1f} (Skipped)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input with threading-specific commands"""
        if super().handle_keyboard_input(key):
            return True
        
        if key == ord('r'):  # Reset
            self.cursor_controller.reset_state()
            print("🔄 System reset!")
        elif key == ord('1'):  # Reduce frame skipping
            self.skip_frames = max(1, self.skip_frames - 1)
            print(f"🎯 Frame skip: {self.skip_frames}")
        elif key == ord('2'):  # Increase frame skipping
            self.skip_frames = min(5, self.skip_frames + 1)
            print(f"🎯 Frame skip: {self.skip_frames}")
        
        return False
    
    def run(self):
        """Run with threading enabled"""
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        
        try:
            # Run main loop
            super().run()
        finally:
            # Stop threading
            self.stop_event.set()
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            print("🧵 Threading stopped")


def main():
    """Main entry point for threaded hand tracking"""
    print("🚀 THREADED Hand Tracking System")
    print("🧵 Multi-threaded processing for ultimate performance")
    
    try:
        tracker = ThreadedHandTracker()
        tracker.run()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please ensure hand_landmarker.task is in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
