"""
Cursor control with improved pinch logic
"""
import time
from typing import Optional, Tuple
from .models import HandInfo
from .system_controller import SystemController


class CursorController:
    """Enhanced cursor control with improved pinch detection - supports both hands"""
    
    def __init__(self, system_controller: SystemController):
        self.system = system_controller
        
        # Hand preference for cursor control (right hand is more natural for most users)
        self.preferred_hand = "Right"  # Right hand default for natural mouse replacement
        self.active_hand = None  # Track which hand is currently controlling cursor
        
        # Sensitivity controls - Apple Vision Pro style
        self.cursor_sensitivity = 1.5  # Base sensitivity multiplier
        self.precision_sensitivity = 0.3  # Precision mode sensitivity
        self.scroll_sensitivity = 2.0  # Scroll gesture sensitivity
        self.click_hold_threshold = 0.8  # Time in seconds for click and hold
        self.precision_mode = False  # Precision mode toggle
        
        # Cursor control state (per hand)
        self.cursor_enabled = False
        self.is_dragging = False  # Track drag state for click and hold
        self.drag_start_pos = None  # Starting position for drag operations
        self.hand_states = {
            "Left": {
                "last_pinch_state": False,
                "pinch_start_time": 0,
                "pinch_release_time": 0,
                "waiting_for_release": False,
                "last_pinch_release_time": 0,
                "is_holding": False  # Track click and hold state
            },
            "Right": {
                "last_pinch_state": False,
                "pinch_start_time": 0,
                "pinch_release_time": 0,
                "waiting_for_release": False,
                "last_pinch_release_time": 0,
                "is_holding": False  # Track click and hold state
            }
        }
        
        # Double pinch detection
        self.double_pinch_threshold = 0.5  # 500ms for double pinch
        
        # Screen dimensions
        self.screen_width, self.screen_height = 1920, 1080  # Will be updated
        try:
            import pyautogui
            self.screen_width, self.screen_height = pyautogui.size()
        except:
            pass
        
        # Smoothing for cursor movement
        self.cursor_smoothing = 0.7  # Higher = more smoothing
        self.last_cursor_pos: Optional[Tuple[float, float]] = None
        
    def handle_cursor_control(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Enhanced cursor control with improved pinch logic - prioritizes right hand"""
        # Only handle cursor control for the preferred hand or active hand
        if hand_info.hand_type != self.preferred_hand and hand_info.hand_type != self.active_hand:
            return
            
        current_time = time.time()
        is_pinching = hand_info.is_pinching
        hand_type = hand_info.hand_type
        
        # Get state for this specific hand
        hand_state = self.hand_states[hand_type]
        last_pinch_state = hand_state["last_pinch_state"]
        
        # Handle pinch state changes
        if is_pinching and not last_pinch_state:
            # Pinch started
            hand_state["pinch_start_time"] = current_time
            
            if self.cursor_enabled:
                # Check for double pinch (left click)
                time_since_last_release = current_time - hand_state["last_pinch_release_time"]
                if time_since_last_release < self.double_pinch_threshold:
                    # Double pinch detected - left click
                    self.system.left_click()
                    print("🖱️  Left click")
                    hand_state["last_pinch_release_time"] = 0  # Reset to prevent triple clicks
            
        elif not is_pinching and last_pinch_state:
            # Pinch ended (released)
            hand_state["pinch_release_time"] = current_time
            pinch_duration = current_time - hand_state["pinch_start_time"]
            
            # Only toggle cursor if pinch was held for a reasonable duration
            if pinch_duration > 0.1:  # At least 100ms pinch
                if not self.cursor_enabled:
                    # Enable cursor tracking
                    self.cursor_enabled = True
                    hand_state["waiting_for_release"] = False
                    print(f"🖱️  Cursor tracking enabled ({hand_type} hand)")
                else:
                    # Disable cursor tracking
                    self.cursor_enabled = False
                    print(f"🖱️  Cursor tracking disabled ({hand_type} hand)")
                
                hand_state["last_pinch_release_time"] = current_time
        
        # Move cursor if enabled and not currently pinching
        if self.cursor_enabled and not is_pinching and not hand_state["waiting_for_release"]:
            self._move_cursor_with_hand(hand_info, frame_width, frame_height)
        
        hand_state["last_pinch_state"] = is_pinching
    
    def handle_multi_hand_cursor_control(self, hands_info: list, frame_width: int, frame_height: int):
        """Handle cursor control with multiple hands, preventing interference"""
        if not hands_info:
            # Reset active hand when no hands are detected
            if self.active_hand is not None:
                self.active_hand = None
            return
            
        # Find preferred hand (left) or fallback to available hand
        preferred_hand = next((h for h in hands_info if h.hand_type == self.preferred_hand), None)
        fallback_hand = hands_info[0] if not preferred_hand else None
        
        # Use preferred hand if available, otherwise use the available hand
        control_hand = preferred_hand or fallback_hand
        
        if control_hand:
            # Only show switching message if the active hand actually changed
            if self.active_hand != control_hand.hand_type:
                if control_hand.hand_type != self.preferred_hand:
                    print(f"🔄 Switching cursor control to {control_hand.hand_type} hand (preferred hand not available)")
                else:
                    print(f"✅ Using preferred {control_hand.hand_type} hand for cursor control")
                self.active_hand = control_hand.hand_type
            
            self.handle_cursor_control(control_hand, frame_width, frame_height)
    
    def set_preferred_hand(self, hand_type: str):
        """Set which hand should be used for cursor control"""
        if hand_type in ["Left", "Right"]:
            self.preferred_hand = hand_type
            print(f"🖱️  Cursor control preference set to {hand_type} hand")
        else:
            print(f"❌ Invalid hand type: {hand_type}. Use 'Left' or 'Right'.")
    
    def set_sensitivity(self, sensitivity_type: str, value: float):
        """Set sensitivity for different cursor operations"""
        if sensitivity_type == "cursor":
            self.cursor_sensitivity = max(0.1, min(5.0, value))
            print(f"🎯 Cursor sensitivity set to {self.cursor_sensitivity:.1f}")
        elif sensitivity_type == "precision":
            self.precision_sensitivity = max(0.05, min(1.0, value))
            print(f"🎯 Precision sensitivity set to {self.precision_sensitivity:.1f}")
        elif sensitivity_type == "scroll":
            self.scroll_sensitivity = max(0.1, min(10.0, value))
            print(f"🎯 Scroll sensitivity set to {self.scroll_sensitivity:.1f}")
        else:
            print(f"❌ Invalid sensitivity type: {sensitivity_type}")
    
    def toggle_precision_mode(self):
        """Toggle precision mode for fine cursor control"""
        self.precision_mode = not self.precision_mode
        mode_text = "ON" if self.precision_mode else "OFF"
        print(f"🎯 Precision mode: {mode_text}")
        return self.precision_mode
    
    def handle_click_and_hold(self, hand_info: HandInfo, current_time: float):
        """Handle click and hold operations for drag and drop"""
        hand_type = hand_info.hand_type
        hand_state = self.hand_states[hand_type]
        is_pinching = hand_info.is_pinching
        
        # Check for click and hold
        if is_pinching and hand_state["pinch_start_time"] > 0:
            pinch_duration = current_time - hand_state["pinch_start_time"]
            
            if pinch_duration >= self.click_hold_threshold and not hand_state["is_holding"]:
                # Start click and hold (drag operation)
                hand_state["is_holding"] = True
                self.is_dragging = True
                self.drag_start_pos = hand_info.fingers["index"].tip_position
                self.system.mouse_down()  # Start drag
                print("🖱️  Click and hold started - drag mode active")
                
        elif not is_pinching and hand_state["is_holding"]:
            # End click and hold (release drag)
            hand_state["is_holding"] = False
            self.is_dragging = False
            self.drag_start_pos = None
            self.system.mouse_up()  # End drag
            print("🖱️  Click and hold released - drag ended")

    def _move_cursor_with_hand(self, hand_info: HandInfo, frame_width: int, frame_height: int):
        """Move cursor based on index finger position with sensitivity control"""
        # Use index finger tip for cursor position
        index_tip = hand_info.fingers["index"].tip_position
        
        # Apply sensitivity based on mode
        active_sensitivity = self.precision_sensitivity if self.precision_mode else self.cursor_sensitivity
        
        # Convert hand position to screen coordinates with sensitivity
        screen_x = index_tip[0] / frame_width * self.screen_width * active_sensitivity
        screen_y = index_tip[1] / frame_height * self.screen_height * active_sensitivity
        
        # Center the movement if sensitivity is not 1.0
        if active_sensitivity != 1.0:
            center_x, center_y = self.screen_width / 2, self.screen_height / 2
            screen_x = center_x + (screen_x - center_x)
            screen_y = center_y + (screen_y - center_y)
        
        # Apply smoothing
        if self.last_cursor_pos is not None:
            smoothing = 0.8 if self.precision_mode else self.cursor_smoothing
            screen_x = self.last_cursor_pos[0] * smoothing + screen_x * (1 - smoothing)
            screen_y = self.last_cursor_pos[1] * smoothing + screen_y * (1 - smoothing)
        
        # Constrain to screen bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        # Move cursor (only if not in drag mode or apply drag movement)
        if self.is_dragging:
            # Continue drag movement
            self.system.move_cursor_while_dragging(screen_x, screen_y)
        else:
            self.system.move_cursor(screen_x, screen_y)
            
        self.last_cursor_pos = (screen_x, screen_y)
    
    def get_status(self) -> str:
        """Get current cursor control status"""
        hand_state = self.hand_states.get(self.preferred_hand, {"waiting_for_release": False})
        if hand_state["waiting_for_release"]:
            return f"WAITING_FOR_RELEASE ({self.preferred_hand})"
        elif self.cursor_enabled:
            return f"ENABLED ({self.preferred_hand})"
        else:
            return f"DISABLED ({self.preferred_hand})"
