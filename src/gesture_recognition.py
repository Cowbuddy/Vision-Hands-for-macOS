"""
Gesture recognition and transition detection
"""
import time
from typing import Dict, List, Optional, Deque
from collections import deque
from .models import FingerState, GestureTransition


class GestureRecognizer:
    """Advanced gesture recognition with transition detection"""
    
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.gesture_history: Deque[str] = deque(maxlen=history_size)
        self.last_gesture_time = time.time()
        self.transition_cooldown = 0.5  # Minimum time between transitions
        
    def recognize_gesture(self, fingers: Dict[str, FingerState], is_fist: bool, is_pinching: bool) -> str:
        """Enhanced gesture recognition"""
        if is_pinching:
            return "pinch"
        
        if is_fist:
            return "fist"
        
        extended = [name for name, finger in fingers.items() if finger.is_extended]
        extended_count = len(extended)
        
        if extended_count == 0:
            return "fist"
        elif extended_count == 1:
            if "index" in extended:
                return "pointing"
            elif "thumb" in extended:
                return "thumbs_up"
            else:
                return "one_finger"
        elif extended_count == 2:
            if "index" in extended and "middle" in extended:
                return "peace_sign"
            elif "thumb" in extended and "index" in extended:
                return "gun"
            else:
                return "two_fingers"
        elif extended_count == 5:
            return "open_hand"
        else:
            return f"{extended_count}_fingers"
    
    def detect_transition(self, current_gesture: str) -> Optional[GestureTransition]:
        """Detect significant gesture transitions"""
        current_time = time.time()
        
        # Only process if enough time has passed and we have history
        if (current_time - self.last_gesture_time < self.transition_cooldown or 
            len(self.gesture_history) == 0):
            self.gesture_history.append(current_gesture)
            return None
        
        last_gesture = self.gesture_history[-1] if self.gesture_history else None
        
        # Detect meaningful transitions
        if last_gesture and last_gesture != current_gesture:
            # Fist to open hand transition (Mission Control trigger)
            if last_gesture == "fist" and current_gesture == "open_hand":
                transition = GestureTransition(
                    from_gesture=last_gesture,
                    to_gesture=current_gesture,
                    timestamp=current_time,
                    confidence=1.0
                )
                self.last_gesture_time = current_time
                self.gesture_history.append(current_gesture)
                return transition
        
        self.gesture_history.append(current_gesture)
        return None
    
    def get_gesture_stability(self, gesture: str, frames: int = 5) -> float:
        """Get stability score for a gesture over recent frames"""
        if len(self.gesture_history) < frames:
            return 0.0
        
        recent_gestures = list(self.gesture_history)[-frames:]
        matches = sum(1 for g in recent_gestures if g == gesture)
        return matches / frames
