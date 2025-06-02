"""
Data structures for hand tracking
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FingerState:
    """Represents the state of a finger"""
    name: str
    is_extended: bool
    tip_position: Tuple[float, float]
    confidence: float = 1.0


@dataclass
class HandInfo:
    """Complete information about a detected hand"""
    hand_type: str  # "Left" or "Right"
    confidence: float
    center_position: Tuple[float, float]
    fingers: Dict[str, FingerState]
    landmarks: List[Tuple[float, float]]
    is_fist: bool
    gesture_name: str
    is_pinching: bool = False  # New field for pinch detection
    pinch_distance: float = 0.0  # Distance between thumb and index finger


@dataclass
class GestureTransition:
    """Represents a transition between gestures"""
    from_gesture: str
    to_gesture: str
    timestamp: float
    confidence: float = 1.0


class HandLandmarks:
    """MediaPipe hand landmark indices"""
    WRIST = 0
    
    # Thumb
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    
    # Index finger
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    
    # Middle finger
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    
    # Ring finger
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    
    # Pinky
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
