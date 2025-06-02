"""
Base hand tracker class to consolidate common functionality
"""
import cv2
import time
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Import MediaPipe
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode
)

from .models import HandInfo
from .hand_analyzer import HandAnalyzer
from .system_controller import SystemController


class BaseHandTracker(ABC):
    """Base class for hand tracking implementations with common functionality"""
    
    def __init__(self, model_path: str = "hand_landmarker.task"):
        self.model_path = model_path
        self.hand_analyzer = HandAnalyzer()
        self.system_controller = SystemController()
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Check model file
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    @abstractmethod
    def create_hand_landmarker(self) -> HandLandmarker:
        """Create hand landmarker with specific configuration"""
        pass
    
    @abstractmethod
    def get_camera_config(self) -> Dict[str, int]:
        """Get camera configuration (width, height, fps)"""
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame - implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_window_title(self) -> str:
        """Get window title for display"""
        pass
    
    def setup_camera(self) -> cv2.VideoCapture:
        """Setup camera with configuration"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open camera")
        
        config = self.get_camera_config()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        cap.set(cv2.CAP_PROP_FPS, config['fps'])
        
        # Minimize buffer for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap
    
    def update_fps(self) -> float:
        """Update FPS calculation"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        return self.current_fps
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle common keyboard inputs. Returns True if should quit."""
        if key == 27:  # ESC
            return True
        return False
    
    def run(self):
        """Main application loop - common structure"""
        self.print_startup_info()
        
        try:
            cap = self.setup_camera()
            print("✅ Camera initialized. Starting hand tracking...")
            
            self.print_instructions()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame (implemented by subclass)
                processed_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow(self.get_window_title(), processed_frame)
                
                # Handle keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if self.handle_keyboard_input(key):
                    break
                
                # Update frame count and FPS
                self.frame_count += 1
                self.update_fps()
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping hand tracking...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup complete!")
    
    @abstractmethod
    def print_startup_info(self):
        """Print startup information specific to implementation"""
        pass
    
    @abstractmethod
    def print_instructions(self):
        """Print usage instructions specific to implementation"""
        pass
    
    def create_base_landmarker_options(self, confidence: float = 0.7, cpu_delegate: bool = False) -> HandLandmarkerOptions:
        """Create base landmarker options"""
        delegate = BaseOptions.Delegate.CPU if cpu_delegate else BaseOptions.Delegate.GPU
        
        base_opts = BaseOptions(
            model_asset_path=self.model_path,
            delegate=delegate
        )
        
        return HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence
        )
    
    def draw_fps(self, frame: np.ndarray, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Draw FPS counter on frame"""
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
