"""
Configuration file for Hand Tracking System
Customize these settings to adjust behavior and performance
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple
import json
import os


@dataclass
class CameraConfig:
    """Camera settings"""
    width: int = 640
    height: int = 480
    fps: int = 60
    buffer_size: int = 1
    device_id: int = 0  # Camera device ID (0 for default)


@dataclass
class HandDetectionConfig:
    """MediaPipe hand detection settings"""
    num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_path: str = "hand_landmarker.task"


@dataclass
class CursorConfig:
    """Cursor control settings"""
    enabled_by_default: bool = True
    default_sensitivity: float = 0.4
    movement_threshold: float = 1.0
    ema_alpha: float = 0.5  # Exponential moving average smoothing factor (0.1-0.9)
    screen_edge_margin: int = 10  # Pixels from screen edge to prevent cursor escaping


@dataclass
class GestureConfig:
    """Gesture recognition settings"""
    pinch_threshold: float = 50.0  # Distance in pixels for pinch detection
    pinch_hold_duration: float = 0.8  # Seconds to hold pinch for drag mode
    gesture_stability_frames: int = 5  # Frames needed for stable gesture
    gesture_stability_score: float = 0.8  # Confidence threshold for gestures
    
    # Scrolling settings
    scroll_sensitivity: int = 3  # Scroll amount per gesture
    scroll_cooldown: float = 0.3  # Seconds between scroll actions


@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    target_fps: int = 60
    gc_interval: int = 300  # Garbage collection every N frames
    fps_history_size: int = 30  # Number of frames to track for FPS calculation
    enable_debug_output: bool = False  # Print debug info every N frames
    debug_interval: int = 60  # Debug output interval in frames


@dataclass
class UIConfig:
    """UI display settings"""
    show_fps: bool = True
    show_hand_landmarks: bool = True
    show_fingertips_only: bool = True  # Only show fingertips (performance mode)
    show_gesture_name: bool = True
    show_mode_info: bool = True
    
    # Colors (BGR format)
    fps_color: Tuple[int, int, int] = (0, 255, 0)
    landmark_color: Tuple[int, int, int] = (255, 0, 0)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    
    font_scale: float = 0.7
    font_thickness: int = 2


@dataclass
class SystemConfig:
    """Complete system configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_detection: HandDetectionConfig = field(default_factory=HandDetectionConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "SystemConfig":
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            config = cls()
            
            # Load each section
            if 'camera' in data:
                config.camera = CameraConfig(**data['camera'])
            if 'hand_detection' in data:
                config.hand_detection = HandDetectionConfig(**data['hand_detection'])
            if 'cursor' in data:
                config.cursor = CursorConfig(**data['cursor'])
            if 'gesture' in data:
                config.gesture = GestureConfig(**data['gesture'])
            if 'performance' in data:
                config.performance = PerformanceConfig(**data['performance'])
            if 'ui' in data:
                config.ui = UIConfig(**data['ui'])
            
            print(f"✅ Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            print(f"❌ Error loading config: {e}, using defaults")
            return cls()
    
    def save_to_file(self, config_path: str = "config.json"):
        """Save configuration to JSON file"""
        try:
            data = {
                'camera': self.camera.__dict__,
                'hand_detection': self.hand_detection.__dict__,
                'cursor': self.cursor.__dict__,
                'gesture': self.gesture.__dict__,
                'performance': self.performance.__dict__,
                'ui': self.ui.__dict__
            }
            
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Configuration saved to {config_path}")
            
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("📋 HAND TRACKING SYSTEM CONFIGURATION")
        print("="*60)
        
        print("\n🎥 Camera Settings:")
        print(f"  Resolution: {self.camera.width}x{self.camera.height}")
        print(f"  FPS Target: {self.camera.fps}")
        print(f"  Device ID: {self.camera.device_id}")
        
        print("\n🖐️  Hand Detection:")
        print(f"  Max Hands: {self.hand_detection.num_hands}")
        print(f"  Detection Confidence: {self.hand_detection.min_detection_confidence}")
        print(f"  Tracking Confidence: {self.hand_detection.min_tracking_confidence}")
        
        print("\n🖱️  Cursor Control:")
        print(f"  Enabled: {self.cursor.enabled_by_default}")
        print(f"  Sensitivity: {self.cursor.default_sensitivity}")
        print(f"  Movement Threshold: {self.cursor.movement_threshold}px")
        print(f"  Smoothing (EMA Alpha): {self.cursor.ema_alpha}")
        
        print("\n👆 Gestures:")
        print(f"  Pinch Threshold: {self.gesture.pinch_threshold}px")
        print(f"  Pinch Hold Duration: {self.gesture.pinch_hold_duration}s")
        print(f"  Scroll Sensitivity: {self.gesture.scroll_sensitivity}")
        
        print("\n⚡ Performance:")
        print(f"  Target FPS: {self.performance.target_fps}")
        print(f"  GC Interval: {self.performance.gc_interval} frames")
        print(f"  Debug Output: {self.performance.enable_debug_output}")
        
        print("\n🎨 UI Settings:")
        print(f"  Show FPS: {self.ui.show_fps}")
        print(f"  Show Landmarks: {self.ui.show_hand_landmarks}")
        print(f"  Fingertips Only: {self.ui.show_fingertips_only}")
        
        print("="*60 + "\n")


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()


def create_example_config():
    """Create an example config.json file"""
    config = SystemConfig()
    config.save_to_file("config.example.json")
    print("📝 Example configuration saved to config.example.json")


if __name__ == "__main__":
    # Create example config when run directly
    create_example_config()
    
    # Also create a default config
    config = SystemConfig()
    config.print_config()
    config.save_to_file("config.json")