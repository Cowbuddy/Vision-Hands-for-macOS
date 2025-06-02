"""
Enhanced gesture recognition with stability and vim-like control modes
"""
import time
import math
from typing import Dict, List, Optional, Deque, Set
from collections import deque
from .models import FingerState, GestureTransition


class StableGestureRecognizer:
    """Enhanced gesture recognition with stability scoring and vim-like modes"""
    
    def __init__(self, history_size: int = 15):
        self.history_size = history_size
        self.gesture_history: Deque[str] = deque(maxlen=history_size)
        self.gesture_timestamps: Deque[float] = deque(maxlen=history_size)
        self.last_stable_gesture = "unknown"
        self.last_transition_time = time.time()
        
        # Stability requirements
        self.min_stability_frames = 12  # Increased - gesture must be stable for 12 frames
        self.min_stability_score = 0.8  # Increased - 80% of recent frames must match
        self.transition_cooldown = 0.8  # Increased - 800ms between transitions
        self.gesture_timeout = 3.0  # Increased - gesture expires after 3 seconds
        
        # Vim-like control modes
        self.current_mode = "NORMAL"  # NORMAL, CURSOR, COMMAND
        self.mode_history = []
        self.mode_transition_time = time.time()
        
    def recognize_gesture(self, fingers: Dict[str, FingerState], is_fist: bool, is_pinching: bool) -> str:
        """Enhanced gesture recognition with better stability"""
        current_time = time.time()
        
        # Primary gesture detection
        if is_pinching:
            return "pinch"
        
        if is_fist:
            return "fist"
        
        # Analyze extended fingers
        extended = [name for name, finger in fingers.items() if finger.is_extended]
        extended_count = len(extended)
        extended_set = set(extended)
        
        # More specific gesture patterns
        if extended_count == 0:
            return "fist"
        elif extended_count == 1:
            if "index" in extended:
                return "point"
            elif "thumb" in extended:
                return "thumb_up"
            elif "middle" in extended:
                return "middle_finger"
            else:
                return "one_finger"
        elif extended_count == 2:
            if {"index", "middle"} == extended_set:
                return "peace"
            elif {"thumb", "index"} == extended_set:
                return "gun"
            elif {"thumb", "pinky"} == extended_set:
                return "call_me"
            else:
                return "two_fingers"
        elif extended_count == 3:
            if {"index", "middle", "ring"} == extended_set:
                return "three_fingers"
            elif {"thumb", "index", "middle"} == extended_set:
                return "ok_sign"
            else:
                return "three_fingers"
        elif extended_count == 4:
            if "thumb" not in extended:
                return "four_fingers"
            else:
                return "four_fingers"
        elif extended_count == 5:
            return "open_hand"
        else:
            return "unknown"
    
    def get_stable_gesture(self, current_gesture: str) -> Optional[str]:
        """Get gesture only if it's stable enough"""
        current_time = time.time()
        
        # Add to history
        self.gesture_history.append(current_gesture)
        self.gesture_timestamps.append(current_time)
        
        # Remove old entries
        while (self.gesture_timestamps and 
               current_time - self.gesture_timestamps[0] > self.gesture_timeout):
            self.gesture_history.popleft()
            self.gesture_timestamps.popleft()
        
        # Check if we have enough recent frames
        if len(self.gesture_history) < self.min_stability_frames:
            return None
        
        # Calculate stability score
        recent_gestures = list(self.gesture_history)[-self.min_stability_frames:]
        matches = sum(1 for g in recent_gestures if g == current_gesture)
        stability_score = matches / len(recent_gestures)
        
        # Return stable gesture if criteria met
        if stability_score >= self.min_stability_score:
            if self.last_stable_gesture != current_gesture:
                # New stable gesture detected
                if current_time - self.last_transition_time >= self.transition_cooldown:
                    self.last_stable_gesture = current_gesture
                    self.last_transition_time = current_time
                    return current_gesture
            else:
                # Same stable gesture continues
                return current_gesture
        
        return self.last_stable_gesture if self.last_stable_gesture != "unknown" else None
    
    def detect_vim_transition(self, stable_gesture: str, hand_type: str) -> Optional[GestureTransition]:
        """Detect vim-like mode transitions based on stable gestures"""
        if not stable_gesture:
            return None
        
        current_time = time.time()
        
        # Only process mode transitions from specific hand (left for commands, right for cursor)
        if hand_type == "Left":
            # Left hand controls modes (vim command mode)
            return self._detect_command_transitions(stable_gesture, current_time)
        elif hand_type == "Right":
            # Right hand controls cursor
            return self._detect_cursor_transitions(stable_gesture, current_time)
        
        return None
    
    def _detect_command_transitions(self, gesture: str, current_time: float) -> Optional[GestureTransition]:
        """Detect command mode transitions (left hand)"""
        transition = None
        
        if gesture == "fist" and self.current_mode != "COMMAND":
            # Fist = Enter command mode
            transition = GestureTransition(
                from_gesture=self.current_mode,
                to_gesture="COMMAND",
                timestamp=current_time,
                confidence=0.9
            )
            self.current_mode = "COMMAND"
            
        elif gesture == "open_hand" and self.current_mode == "COMMAND":
            # Open hand from command mode = Execute action
            transition = GestureTransition(
                from_gesture="COMMAND",
                to_gesture="EXECUTE",
                timestamp=current_time,
                confidence=0.9
            )
            self.current_mode = "NORMAL"
            
        elif gesture == "peace" and self.current_mode != "NORMAL":
            # Peace sign = Back to normal mode
            transition = GestureTransition(
                from_gesture=self.current_mode,
                to_gesture="NORMAL",
                timestamp=current_time,
                confidence=0.8
            )
            self.current_mode = "NORMAL"
        
        return transition
    
    def _detect_cursor_transitions(self, gesture: str, current_time: float) -> Optional[GestureTransition]:
        """Detect cursor control transitions (right hand)"""
        transition = None
        
        if gesture == "point" and self.current_mode != "CURSOR":
            # Point = Enter cursor mode
            transition = GestureTransition(
                from_gesture=self.current_mode,
                to_gesture="CURSOR",
                timestamp=current_time,
                confidence=0.9
            )
            
        elif gesture == "pinch":
            # Pinch = Click action in cursor mode
            transition = GestureTransition(
                from_gesture="CURSOR",
                to_gesture="CLICK",
                timestamp=current_time,
                confidence=0.9
            )
        
        return transition
    
    def get_gesture_confidence(self, gesture: str) -> float:
        """Get confidence score for current gesture"""
        if len(self.gesture_history) == 0:
            return 0.0
        
        recent_gestures = list(self.gesture_history)[-self.min_stability_frames:]
        matches = sum(1 for g in recent_gestures if g == gesture)
        return matches / len(recent_gestures) if recent_gestures else 0.0
    
    def reset_stability(self):
        """Reset gesture stability (useful when hands disappear)"""
        self.gesture_history.clear()
        self.gesture_timestamps.clear()
        self.last_stable_gesture = "unknown"
    
    def get_mode_info(self) -> Dict:
        """Get current mode information"""
        return {
            "mode": self.current_mode,
            "stable_gesture": self.last_stable_gesture,
            "confidence": self.get_gesture_confidence(self.last_stable_gesture),
            "mode_duration": time.time() - self.mode_transition_time
        }


class GestureFilter:
    """Filter and validate gestures to prevent noise"""
    
    def __init__(self):
        self.noise_gestures = {"unknown", "three_fingers", "four_fingers"}
        self.priority_gestures = {"fist", "open_hand", "pinch", "point", "peace"}
        
    def is_gesture_valid(self, gesture: str, confidence: float) -> bool:
        """Check if gesture is valid and not noise"""
        if gesture in self.noise_gestures and confidence < 0.8:
            return False
        
        if gesture in self.priority_gestures:
            return confidence > 0.6
        
        return confidence > 0.7
    
    def filter_gesture(self, gesture: str, confidence: float) -> Optional[str]:
        """Filter gesture and return cleaned version"""
        if not self.is_gesture_valid(gesture, confidence):
            return None
        return gesture
