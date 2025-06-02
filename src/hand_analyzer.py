"""
Hand analysis with improved finger detection
"""
import math
import numpy as np
from typing import Dict, List
from .models import HandInfo, FingerState, HandLandmarks
from .gesture_recognition import GestureRecognizer


def vec(a, b):
    """Vector from point b to point a"""
    return np.array([a.x - b.x, a.y - b.y, a.z - b.z])


def angle_between_vectors(v1, v2):
    """Calculate angle in degrees between two vectors"""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = np.clip(dot_product / (norms + 1e-6), -1, 1)
    return math.degrees(np.arccos(cos_angle))


def is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    """Improved finger extension detection using multiple methods"""
    # Method 1: Angle analysis (original)
    tip_pip_vec = vec(landmarks[tip_idx], landmarks[pip_idx])
    pip_mcp_vec = vec(landmarks[mcp_idx], landmarks[pip_idx])
    angle = angle_between_vectors(tip_pip_vec, pip_mcp_vec)
    angle_extended = angle < 40  # Slightly more lenient
    
    # Method 2: Distance analysis - tip should be farther from wrist than PIP
    wrist = landmarks[HandLandmarks.WRIST]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    
    tip_wrist_dist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
    pip_wrist_dist = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
    distance_extended = tip_wrist_dist > pip_wrist_dist * 1.1
    
    # Method 3: Y-coordinate analysis (tip should be higher/lower than PIP depending on hand orientation)
    y_extended = abs(tip.y - pip.y) > 0.02  # Some vertical separation
    
    # Combine methods - finger is extended if 2 out of 3 methods agree
    votes = [angle_extended, distance_extended, y_extended]
    return sum(votes) >= 2


def is_thumb_extended(landmarks):
    """Improved thumb detection using multiple criteria"""
    thumb_tip = landmarks[HandLandmarks.THUMB_TIP]
    thumb_ip = landmarks[HandLandmarks.THUMB_IP]
    thumb_mcp = landmarks[HandLandmarks.THUMB_MCP]
    wrist = landmarks[HandLandmarks.WRIST]
    index_mcp = landmarks[HandLandmarks.INDEX_MCP]
    
    # Method 1: Distance from thumb tip to thumb MCP vs IP to MCP
    tip_mcp_dist = math.sqrt((thumb_tip.x - thumb_mcp.x)**2 + (thumb_tip.y - thumb_mcp.y)**2)
    ip_mcp_dist = math.sqrt((thumb_ip.x - thumb_mcp.x)**2 + (thumb_ip.y - thumb_mcp.y)**2)
    distance_ratio = tip_mcp_dist / (ip_mcp_dist + 1e-6)
    distance_extended = distance_ratio > 1.3
    
    # Method 2: Thumb tip distance from index MCP (when extended, thumb is farther)
    tip_index_dist = math.sqrt((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)
    ip_index_dist = math.sqrt((thumb_ip.x - index_mcp.x)**2 + (thumb_ip.y - index_mcp.y)**2)
    index_separation = tip_index_dist > ip_index_dist * 0.9
    
    # Method 3: Angle analysis similar to other fingers
    tip_ip_vec = vec(thumb_tip, thumb_ip)
    ip_mcp_vec = vec(thumb_mcp, thumb_ip)
    angle = angle_between_vectors(tip_ip_vec, ip_mcp_vec)
    angle_extended = angle < 50  # More lenient for thumb
    
    # Combine methods
    votes = [distance_extended, index_separation, angle_extended]
    return sum(votes) >= 2


class HandAnalyzer:
    """Enhanced hand analysis with gesture recognition"""
    
    def __init__(self):
        self.finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        self.gesture_recognizer = GestureRecognizer()
        
    def analyze_hand(self, hand_landmarks, handedness, frame_width, frame_height) -> HandInfo:
        """Analyze a single hand and return comprehensive information"""
        landmarks = hand_landmarks
        
        # Fix left/right hand classification - MediaPipe gives mirrored results
        original_hand_type = handedness[0].category_name
        # Flip the hand classification to correct the mirroring
        hand_type = "Right" if original_hand_type == "Left" else "Left"
        confidence = handedness[0].score
        
        # Convert landmarks to pixel coordinates
        landmark_coords = []
        for lm in landmarks:
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            landmark_coords.append((x, y))
        
        # Calculate hand center (average of all landmarks)
        center_x = sum(coord[0] for coord in landmark_coords) / len(landmark_coords)
        center_y = sum(coord[1] for coord in landmark_coords) / len(landmark_coords)
        
        # Analyze each finger
        fingers = {}
        
        # Thumb (special case)
        thumb_extended = is_thumb_extended(landmarks)
        thumb_tip = landmark_coords[HandLandmarks.THUMB_TIP]
        fingers["thumb"] = FingerState("thumb", thumb_extended, thumb_tip)
        
        # Other fingers
        finger_configs = [
            ("index", HandLandmarks.INDEX_TIP, HandLandmarks.INDEX_PIP, HandLandmarks.INDEX_MCP),
            ("middle", HandLandmarks.MIDDLE_TIP, HandLandmarks.MIDDLE_PIP, HandLandmarks.MIDDLE_MCP),
            ("ring", HandLandmarks.RING_TIP, HandLandmarks.RING_PIP, HandLandmarks.RING_MCP),
            ("pinky", HandLandmarks.PINKY_TIP, HandLandmarks.PINKY_PIP, HandLandmarks.PINKY_MCP)
        ]
        
        for name, tip_idx, pip_idx, mcp_idx in finger_configs:
            is_extended = is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx)
            tip_pos = landmark_coords[tip_idx]
            fingers[name] = FingerState(name, is_extended, tip_pos)
        
        # Determine if it's a fist
        extended_fingers = [f.is_extended for f in fingers.values()]
        is_fist = not any(extended_fingers)
        
        # Calculate pinch detection (thumb and index finger distance)
        thumb_tip_pos = fingers["thumb"].tip_position
        index_tip_pos = fingers["index"].tip_position
        pinch_distance = math.sqrt(
            (thumb_tip_pos[0] - index_tip_pos[0])**2 + 
            (thumb_tip_pos[1] - index_tip_pos[1])**2
        )
        is_pinching = pinch_distance < 50  # Adjust threshold as needed
        
        # Enhanced gesture recognition
        gesture_name = self.gesture_recognizer.recognize_gesture(fingers, is_fist, is_pinching)
        
        return HandInfo(
            hand_type=hand_type,
            confidence=confidence,
            center_position=(center_x, center_y),
            fingers=fingers,
            landmarks=landmark_coords,
            is_fist=is_fist,
            gesture_name=gesture_name,
            is_pinching=is_pinching,
            pinch_distance=pinch_distance
        )
    
    def detect_gesture_transition(self, hand_info: HandInfo):
        """Detect gesture transitions for the current hand"""
        return self.gesture_recognizer.detect_transition(hand_info.gesture_name)
