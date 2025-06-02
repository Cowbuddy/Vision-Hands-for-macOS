"""
Vim-like modal cursor controller optimized for right-hand dominance
"""
import time
import pyautogui
from typing import Optional, Tuple, Dict, Any
from .models import HandInfo
from .system_controller import SystemController


class VimCursorController:
    """Vim-like modal cursor control optimized for right-hand users"""
    
    def __init__(self, system_controller: SystemController):
        self.system = system_controller
        
        # Control modes (vim-like)
        self.mode = "NORMAL"  # NORMAL, CURSOR, PRECISION
        self.cursor_enabled = False
        self.precision_mode = False
        
        # Hand assignments (right-hand dominant)
        self.cursor_hand = "Right"  # Right hand controls cursor
        self.command_hand = "Left"  # Left hand controls modes
        
        # Performance settings
        self.cursor_smoothing = 0.3  # Reduced for responsiveness
        self.precision_smoothing = 0.8  # Higher for precision mode
        self.movement_threshold = 5  # Minimum movement to register
        
        # Click detection
        self.pinch_threshold = 40  # Distance in pixels for pinch detection
        self.double_click_window = 0.4  # 400ms for double click
        self.last_click_time = 0
        self.click_count = 0
        
        # State tracking
        self.last_cursor_pos: Optional[Tuple[float, float]] = None
        self.last_pinch_state = False
        self.pinch_start_time = 0
        self.gesture_hold_time = 0
        
        # Screen bounds
        try:
            self.screen_width, self.screen_height = pyautogui.size()
        except:
            self.screen_width, self.screen_height = 1920, 1080
        
        # Movement zones for precision
        self.precision_zone_size = 0.3  # 30% of screen for precision control
        
    def set_mode(self, new_mode: str):
        """Set cursor control mode"""
        if new_mode in ["NORMAL", "CURSOR", "PRECISION"]:
            old_mode = self.mode
            self.mode = new_mode
            
            if new_mode == "CURSOR":
                self.cursor_enabled = True
                self.precision_mode = False
                print(f"🎯 Cursor mode ENABLED")
            elif new_mode == "PRECISION":
                self.cursor_enabled = True
                self.precision_mode = True
                print(f"🎯 Precision cursor mode ENABLED")
            else:  # NORMAL
                self.cursor_enabled = False
                self.precision_mode = False
                print(f"🎯 Cursor mode DISABLED")
            
            return old_mode != new_mode
        return False
    
    def handle_right_hand_cursor(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Handle cursor control with right hand (dominant hand)"""
        if hand_info.hand_type != self.cursor_hand or not self.cursor_enabled:
            return
        
        current_time = time.time()
        
        # Handle pinch gestures for clicking
        self._handle_pinch_clicks(hand_info, current_time)
        
        # Handle cursor movement based on gesture
        if hand_info.gesture_name == "point":
            # Point gesture = move cursor
            self._move_cursor_with_finger(hand_info, frame_width, frame_height)
        elif hand_info.gesture_name == "fist":
            # Fist = hold position (no movement)
            pass
        elif hand_info.gesture_name == "open_hand":
            # Open hand = free movement with palm center
            self._move_cursor_with_palm(hand_info, frame_width, frame_height)
    
    def handle_left_hand_commands(self, hand_info: HandInfo) -> Optional[str]:
        """Handle mode commands with left hand"""
        if hand_info.hand_type != self.command_hand:
            return None
        
        gesture = hand_info.gesture_name
        current_time = time.time()
        
        # Mode switching gestures
        if gesture == "point":
            # Point = Enter cursor mode
            if self.set_mode("CURSOR"):
                return "CURSOR_MODE"
        elif gesture == "peace":
            # Peace = Enter precision mode
            if self.set_mode("PRECISION"):
                return "PRECISION_MODE"
        elif gesture == "fist":
            # Fist = Return to normal mode
            if self.set_mode("NORMAL"):
                return "NORMAL_MODE"
        elif gesture == "thumb_up":
            # Thumb up = Quick action (could be Mission Control)
            return "QUICK_ACTION"
        elif gesture == "call_me":
            # Call me = Special action
            return "SPECIAL_ACTION"
        
        return None
    
    def _handle_pinch_clicks(self, hand_info: HandInfo, current_time: float):
        """Handle pinch-based clicking"""
        is_pinching = hand_info.is_pinching
        
        # Detect pinch start
        if is_pinching and not self.last_pinch_state:
            self.pinch_start_time = current_time
            
        # Detect pinch release (click)
        elif not is_pinching and self.last_pinch_state:
            pinch_duration = current_time - self.pinch_start_time
            
            # Valid click if pinch was held briefly
            if 0.05 < pinch_duration < 0.5:  # 50ms to 500ms
                self._perform_click(current_time)
        
        self.last_pinch_state = is_pinching
    
    def _perform_click(self, current_time: float):
        """Perform click with double-click detection"""
        time_since_last = current_time - self.last_click_time
        
        if time_since_last < self.double_click_window:
            # Double click detected
            self.click_count += 1
            if self.click_count == 1:  # Second click of double click
                self.system.left_click()  # Second click
                print("🖱️  Double click")
                self.click_count = 0
        else:
            # Single click
            self.system.left_click()
            print("🖱️  Click")
            self.click_count = 0
        
        self.last_click_time = current_time
    
    def _move_cursor_with_finger(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Move cursor using index finger position"""
        if "index" not in hand_info.fingers or not hand_info.fingers["index"].is_extended:
            return
        
        finger_pos = hand_info.fingers["index"].tip_position
        self._move_cursor_to_position(finger_pos, frame_width, frame_height)
    
    def _move_cursor_with_palm(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Move cursor using palm center"""
        palm_pos = hand_info.center_position
        self._move_cursor_to_position(palm_pos, frame_width, frame_height)
    
    def _move_cursor_to_position(self, hand_pos: Tuple[float, float], frame_width: int, frame_height: int):
        """Move cursor to hand position with smoothing"""
        # Convert hand position to screen coordinates
        screen_x = (1.0 - hand_pos[0] / frame_width) * self.screen_width  # Flip X for natural movement
        screen_y = hand_pos[1] / frame_height * self.screen_height
        
        # Apply precision mode if enabled
        if self.precision_mode:
            screen_x, screen_y = self._apply_precision_scaling(screen_x, screen_y)
        
        # Apply smoothing
        smoothing = self.precision_smoothing if self.precision_mode else self.cursor_smoothing
        if self.last_cursor_pos is not None:
            screen_x = self.last_cursor_pos[0] * smoothing + screen_x * (1 - smoothing)
            screen_y = self.last_cursor_pos[1] * smoothing + screen_y * (1 - smoothing)
        
        # Check movement threshold
        if self.last_cursor_pos is not None:
            dx = abs(screen_x - self.last_cursor_pos[0])
            dy = abs(screen_y - self.last_cursor_pos[1])
            if dx < self.movement_threshold and dy < self.movement_threshold:
                return  # Movement too small, ignore
        
        # Constrain to screen bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        # Move cursor
        self.system.move_cursor(screen_x, screen_y)
        self.last_cursor_pos = (screen_x, screen_y)
    
    def _apply_precision_scaling(self, x: float, y: float) -> Tuple[float, float]:
        """Apply precision scaling to cursor movement"""
        # Map hand movement to smaller screen area for precision
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        # Scale movement around current cursor position
        if self.last_cursor_pos:
            center_x, center_y = self.last_cursor_pos
        
        # Reduce movement range
        max_movement = min(self.screen_width, self.screen_height) * self.precision_zone_size
        
        # Calculate relative movement
        rel_x = (x - center_x) * self.precision_zone_size
        rel_y = (y - center_y) * self.precision_zone_size
        
        return center_x + rel_x, center_y + rel_y
    
    def handle_dual_hand_control(self, hands_info: list, frame_width: int, frame_height: int) -> Optional[str]:
        """Handle both hands with clear separation of duties"""
        command_result = None
        
        for hand_info in hands_info:
            if hand_info.hand_type == self.command_hand:
                # Left hand handles mode commands
                command_result = self.handle_left_hand_commands(hand_info)
            elif hand_info.hand_type == self.cursor_hand:
                # Right hand handles cursor movement
                self.handle_right_hand_cursor(hand_info, frame_width, frame_height)
        
        return command_result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status"""
        return {
            "mode": self.mode,
            "cursor_enabled": self.cursor_enabled,
            "precision_mode": self.precision_mode,
            "cursor_hand": self.cursor_hand,
            "command_hand": self.command_hand,
            "cursor_position": self.last_cursor_pos
        }
    
    def reset_state(self):
        """Reset controller state"""
        self.last_cursor_pos = None
        self.last_pinch_state = False
        self.click_count = 0
        self.gesture_hold_time = 0
