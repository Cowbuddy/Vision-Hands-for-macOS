#!/usr/bin/env python3
"""
Custom Gesture Examples for Hand Tracking System

This file demonstrates how to create and integrate custom gestures
into the hand tracking system.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from src.models import HandInfo, FingerState


class CustomGestureRecognizer:
    """Example custom gesture recognizer with additional gestures"""
    
    def __init__(self):
        self.gesture_history = []
        
    def recognize_custom_gesture(self, hand_info: HandInfo) -> str:
        """
        Recognize custom gestures beyond the basic set
        
        Returns:
            Gesture name string
        """
        fingers = hand_info.fingers
        
        # Get extended fingers
        extended = [name for name, finger in fingers.items() if finger.is_extended]
        extended_count = len(extended)
        extended_set = set(extended)
        
        # Custom gesture: "OK Sign" (thumb + index forming circle, others extended)
        if self._is_ok_sign(fingers):
            return "ok_sign"
        
        # Custom gesture: "Call Me" (thumb + pinky extended, others folded)
        if extended_set == {"thumb", "pinky"} and extended_count == 2:
            return "call_me"
        
        # Custom gesture: "Rock On" (index + pinky extended, others folded)
        if extended_set == {"index", "pinky"} and extended_count == 2:
            return "rock_on"
        
        # Custom gesture: "Thumbs Up"
        if extended_set == {"thumb"} and extended_count == 1:
            return "thumbs_up"
        
        # Custom gesture: "Thumbs Down" (requires checking thumb position)
        if self._is_thumbs_down(fingers):
            return "thumbs_down"
        
        # Custom gesture: "Gun" (thumb + index extended, forming gun shape)
        if extended_set == {"thumb", "index"} and extended_count == 2:
            return "gun"
        
        # Custom gesture: "Swipe Left" (requires motion tracking)
        if self._is_swipe_gesture(hand_info, "left"):
            return "swipe_left"
        
        # Custom gesture: "Swipe Right"
        if self._is_swipe_gesture(hand_info, "right"):
            return "swipe_right"
        
        return "unknown"
    
    def _is_ok_sign(self, fingers: Dict[str, FingerState]) -> bool:
        """
        Detect OK sign: thumb and index finger tips close together,
        other fingers extended
        """
        thumb_pos = fingers["thumb"].tip_position
        index_pos = fingers["index"].tip_position
        
        # Check if thumb and index are close
        distance = ((thumb_pos[0] - index_pos[0])**2 + 
                   (thumb_pos[1] - index_pos[1])**2)**0.5
        
        tips_close = distance < 40  # Adjust threshold as needed
        
        # Check if other fingers are extended
        other_extended = (fingers["middle"].is_extended and 
                         fingers["ring"].is_extended and 
                         fingers["pinky"].is_extended)
        
        return tips_close and other_extended
    
    def _is_thumbs_down(self, fingers: Dict[str, FingerState]) -> bool:
        """
        Detect thumbs down: only thumb extended, pointing downward
        """
        if not fingers["thumb"].is_extended:
            return False
        
        # Check if other fingers are folded
        others_folded = not any([
            fingers["index"].is_extended,
            fingers["middle"].is_extended,
            fingers["ring"].is_extended,
            fingers["pinky"].is_extended
        ])
        
        if not others_folded:
            return False
        
        # Check if thumb is pointing down (y-coordinate check)
        # This is a simplified check - you might need more sophisticated logic
        thumb_tip = fingers["thumb"].tip_position
        thumb_base_y = thumb_tip[1]  # Approximate
        
        # If thumb tip is below base, it's pointing down
        # Note: In image coordinates, y increases downward
        return True  # Simplified - enhance with proper orientation check
    
    def _is_swipe_gesture(self, hand_info: HandInfo, direction: str) -> bool:
        """
        Detect swipe gesture (requires tracking hand motion over time)
        
        This is a placeholder - implement with position history tracking
        """
        # Store hand center position
        current_pos = hand_info.center_position
        self.gesture_history.append(current_pos)
        
        # Keep only recent history (last 10 frames)
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)
        
        # Need at least 5 frames to detect swipe
        if len(self.gesture_history) < 5:
            return False
        
        # Calculate horizontal movement
        start_x = self.gesture_history[0][0]
        end_x = self.gesture_history[-1][0]
        movement = end_x - start_x
        
        # Detect swipe based on direction and threshold
        threshold = 100  # pixels
        
        if direction == "left":
            return movement < -threshold
        elif direction == "right":
            return movement > threshold
        
        return False


class GestureActionMapper:
    """Map gestures to system actions"""
    
    def __init__(self):
        self.action_map = {
            # Basic gestures
            "open_hand": self.action_show_desktop,
            "fist": self.action_minimize_window,
            "peace": self.action_mission_control,
            
            # Custom gestures
            "ok_sign": self.action_screenshot,
            "call_me": self.action_open_terminal,
            "rock_on": self.action_play_pause,
            "thumbs_up": self.action_volume_up,
            "thumbs_down": self.action_volume_down,
            "gun": self.action_next_track,
            "swipe_left": self.action_prev_workspace,
            "swipe_right": self.action_next_workspace,
        }
    
    def execute_gesture(self, gesture_name: str):
        """Execute action for given gesture"""
        if gesture_name in self.action_map:
            action = self.action_map[gesture_name]
            action()
            print(f"✨ Executed: {gesture_name}")
        else:
            print(f"❓ Unknown gesture: {gesture_name}")
    
    # Action implementations
    def action_show_desktop(self):
        """Show desktop (F11)"""
        import pyautogui
        pyautogui.press('f11')
        print("🖥️  Desktop shown")
    
    def action_minimize_window(self):
        """Minimize current window (Cmd+M)"""
        import pyautogui
        pyautogui.hotkey('command', 'm')
        print("📦 Window minimized")
    
    def action_mission_control(self):
        """Trigger Mission Control"""
        import pyautogui
        pyautogui.press('f3')
        print("🚀 Mission Control")
    
    def action_screenshot(self):
        """Take screenshot (Cmd+Shift+4)"""
        import pyautogui
        pyautogui.hotkey('command', 'shift', '4')
        print("📸 Screenshot mode")
    
    def action_open_terminal(self):
        """Open Terminal (Cmd+Space, type terminal)"""
        import pyautogui
        pyautogui.hotkey('command', 'space')
        import time
        time.sleep(0.3)
        pyautogui.typewrite('terminal', interval=0.05)
        pyautogui.press('enter')
        print("💻 Opening Terminal")
    
    def action_play_pause(self):
        """Play/pause media"""
        import pyautogui
        pyautogui.press('playpause')
        print("⏯️  Play/Pause")
    
    def action_volume_up(self):
        """Increase volume"""
        import pyautogui
        pyautogui.press('volumeup')
        print("🔊 Volume up")
    
    def action_volume_down(self):
        """Decrease volume"""
        import pyautogui
        pyautogui.press('volumedown')
        print("🔉 Volume down")
    
    def action_next_track(self):
        """Next media track"""
        import pyautogui
        pyautogui.press('nexttrack')
        print("⏭️  Next track")
    
    def action_prev_workspace(self):
        """Previous workspace (Ctrl+Left)"""
        import pyautogui
        pyautogui.hotkey('ctrl', 'left')
        print("⬅️  Previous workspace")
    
    def action_next_workspace(self):
        """Next workspace (Ctrl+Right)"""
        import pyautogui
        pyautogui.hotkey('ctrl', 'right')
        print("➡️  Next workspace")


def example_integration():
    """
    Example of how to integrate custom gestures into the main application
    """
    print("="*60)
    print("CUSTOM GESTURE INTEGRATION EXAMPLE")
    print("="*60)
    print("\nTo integrate these gestures into your main application:\n")
    
    print("1. Import the custom recognizer:")
    print("   from examples.custom_gestures import CustomGestureRecognizer")
    print()
    
    print("2. Initialize it in your tracker:")
    print("   self.custom_recognizer = CustomGestureRecognizer()")
    print()
    
    print("3. Use it in hand processing:")
    print("   custom_gesture = self.custom_recognizer.recognize_custom_gesture(hand_info)")
    print("   if custom_gesture != 'unknown':")
    print("       print(f'Custom gesture detected: {custom_gesture}')")
    print()
    
    print("4. Map gestures to actions:")
    print("   self.action_mapper = GestureActionMapper()")
    print("   self.action_mapper.execute_gesture(custom_gesture)")
    print()
    
    print("="*60)
    print("\nAvailable Custom Gestures:")
    print("-"*60)
    gestures = [
        ("OK Sign", "Thumb + index circle, others extended", "Screenshot"),
        ("Call Me", "Thumb + pinky extended", "Open Terminal"),
        ("Rock On", "Index + pinky extended", "Play/Pause media"),
        ("Thumbs Up", "Only thumb extended upward", "Volume up"),
        ("Thumbs Down", "Only thumb extended downward", "Volume down"),
        ("Gun", "Thumb + index extended", "Next track"),
        ("Swipe Left", "Hand moving left", "Previous workspace"),
        ("Swipe Right", "Hand moving right", "Next workspace"),
    ]
    
    for name, description, action in gestures:
        print(f"  • {name:15} - {description:35} → {action}")
    
    print("="*60)


if __name__ == "__main__":
    example_integration()
    
    print("\n💡 Tips for Creating Custom Gestures:")
    print("-"*60)
    print("1. Use finger extension states from hand_info.fingers")
    print("2. Calculate distances between finger tips for pinch-like gestures")
    print("3. Track hand position over time for motion gestures")
    print("4. Use gesture history for stability")
    print("5. Test thoroughly in different lighting conditions")
    print("6. Add cooldowns to prevent gesture spam")
    print("="*60)