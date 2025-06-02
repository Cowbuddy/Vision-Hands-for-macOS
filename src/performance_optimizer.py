"""
Performance optimization utilities for hand tracking
"""
import cv2
import numpy as np
import time
from typing import Optional, Tuple


class FrameOptimizer:
    """Optimizes frame processing for higher FPS"""
    
    def __init__(self):
        self.frame_skip = 0  # Skip frames for performance
        self.frame_counter = 0
        self.target_fps = 60
        self.last_process_time = time.time()
        
        # Frame processing optimization
        self.resize_factor = 0.8  # Resize frames for faster processing
        self.process_every_n_frames = 1  # Process every N frames
        
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed"""
        self.frame_counter += 1
        return self.frame_counter % self.process_every_n_frames == 0
    
    def optimize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize frame for faster processing"""
        start_time = time.time()
        
        # Resize for faster processing if needed
        if self.resize_factor < 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB (faster than RGBA)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processing_time = time.time() - start_time
        return rgb_frame, processing_time
    
    def restore_frame_scale(self, frame: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Restore frame to original size"""
        if self.resize_factor < 1.0:
            return cv2.resize(frame, (original_shape[1], original_shape[0]))
        return frame
    
    def adjust_performance(self, current_fps: float):
        """Dynamically adjust performance settings based on current FPS"""
        if current_fps < self.target_fps * 0.8:  # If FPS is too low
            if self.process_every_n_frames < 3:
                self.process_every_n_frames += 1
            elif self.resize_factor > 0.6:
                self.resize_factor -= 0.1
        elif current_fps > self.target_fps * 1.2:  # If FPS is too high, increase quality
            if self.process_every_n_frames > 1:
                self.process_every_n_frames -= 1
            elif self.resize_factor < 1.0:
                self.resize_factor += 0.1


class VisualizationOptimizer:
    """Optimizes visualization rendering for better performance"""
    
    def __init__(self):
        self.draw_landmarks = True
        self.draw_connections = True
        self.draw_detailed_info = True
        self.font_scale = 0.6
        self.line_thickness = 2
        
    def draw_minimal_hand(self, frame: np.ndarray, hand_info, color: Tuple[int, int, int]):
        """Draw minimal hand visualization for performance"""
        # Only draw key landmarks
        key_points = [4, 8, 12, 16, 20]  # Fingertips
        for idx in key_points:
            if idx < len(hand_info.landmarks):
                x, y = hand_info.landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    
    def draw_gesture_indicator(self, frame: np.ndarray, hand_info, color: Tuple[int, int, int]):
        """Draw simple gesture indicator"""
        center_x, center_y = map(int, hand_info.center_position)
        
        # Simple gesture text
        gesture_text = f"{hand_info.hand_type[0]}: {hand_info.gesture_name}"
        cv2.putText(frame, gesture_text, (center_x - 40, center_y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.line_thickness)
    
    def toggle_detail_level(self):
        """Toggle between detailed and minimal visualization"""
        self.draw_detailed_info = not self.draw_detailed_info
        if not self.draw_detailed_info:
            self.draw_connections = False
            self.font_scale = 0.5
            self.line_thickness = 1
        else:
            self.draw_connections = True
            self.font_scale = 0.6
            self.line_thickness = 2


class FPSMonitor:
    """Enhanced FPS monitoring and optimization"""
    
    def __init__(self, target_fps: float = 60):
        self.target_fps = target_fps
        self.fps_history = []
        self.frame_times = []
        self.last_time = time.time()
        self.frame_count = 0
        
    def update(self) -> float:
        """Update FPS calculation"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time
        self.frame_count += 1
        
        # Keep only recent frame times
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate average FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.fps_history.append(fps)
            
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            
            return fps
        return 0
    
    def get_average_fps(self) -> float:
        """Get smoothed average FPS"""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
    
    def is_performance_good(self) -> bool:
        """Check if performance is meeting targets"""
        avg_fps = self.get_average_fps()
        return avg_fps >= self.target_fps * 0.8
    
    def get_performance_status(self) -> str:
        """Get performance status string"""
        avg_fps = self.get_average_fps()
        if avg_fps >= self.target_fps * 0.9:
            return "EXCELLENT"
        elif avg_fps >= self.target_fps * 0.7:
            return "GOOD"
        elif avg_fps >= self.target_fps * 0.5:
            return "FAIR"
        else:
            return "POOR"
