"""Camera-free checks: load tracker logic without importing hardware libraries."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / 'main.py').read_text())
tracker_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'OptimizedHandTracker')
namespace = dict(HandInfo=object, json=json, time=time)
exec(compile(ast.Module(body=[tracker_class], type_ignores=[]), 'main.py', 'exec'), namespace)


class GestureBoundaryTests(unittest.TestCase):
    def tracker(self, events_only=False):
        tracker = namespace['OptimizedHandTracker'].__new__(namespace['OptimizedHandTracker'])
        tracker.events_only = events_only
        tracker.event_stream = io.StringIO()
        tracker.visible_hands = set()
        tracker.cursor_enabled = True
        tracker.frame_count = 1
        tracker.gesture_states = dict(pinch_held=False, pinch_start_time=0,
                                     drag_active=False, last_scroll_time=0,
                                     left_pinch_held=False)
        tracker.system_controller = Mock()
        tracker.hand_analyzer = Mock()
        tracker.gesture_recognizer = Mock()
        tracker.scroll_cooldown = .5
        tracker.scroll_sensitivity = 3
        return tracker

    def hand(self, pinch=False, gesture='point', side='Right'):
        return SimpleNamespace(is_pinching=pinch, gesture_name=gesture, hand_type=side)

    def test_events_only_never_calls_os_for_click_drag_scroll_or_movement(self):
        tracker = self.tracker(True)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(time, 'time', return_value=10):
            tracker._handle_right_hand_cursor(self.hand(True), 640, 480)
            tracker._handle_right_hand_cursor(self.hand(True), 640, 480)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(time, 'time', return_value=11):
            tracker._handle_right_hand_cursor(self.hand(True), 640, 480)
            tracker._handle_right_hand_cursor(self.hand(False), 640, 480)
            tracker._handle_right_hand_cursor(self.hand(gesture='three'), 640, 480)
        self.assertEqual(tracker.system_controller.mock_calls, [])
        events = [json.loads(line) for line in tracker.event_stream.getvalue().splitlines()]
        self.assertEqual([e['type'] for e in events], ['pinch.start', 'pinch.end'])
        self.assertEqual(events[0], dict(version=1, source='visionhands', type='pinch.start', hand='Right', timestamp=10))

    def test_tracking_loss_releases_drag_once_without_click(self):
        tracker = self.tracker()
        tracker.visible_hands = {'Right'}
        tracker.gesture_states.update(drag_active=True, pinch_held=True)
        tracker._process_hands_optimized([], [], 640, 480)
        tracker._process_hands_optimized([], [], 640, 480)
        tracker.system_controller.mouse_up.assert_called_once_with()
        tracker.system_controller.left_click.assert_not_called()
        self.assertFalse(tracker.gesture_states['drag_active'])

    def test_loss_event_emits_once_and_never_releases_os_in_event_mode(self):
        tracker = self.tracker(True)
        tracker.visible_hands = {'Right'}
        tracker.gesture_states.update(drag_active=True, pinch_held=True)
        tracker._process_hands_optimized([], [], 640, 480)
        tracker._process_hands_optimized([], [], 640, 480)
        self.assertEqual(tracker.system_controller.mock_calls, [])
        events = [json.loads(line) for line in tracker.event_stream.getvalue().splitlines()]
        self.assertEqual([e['type'] for e in events], ['tracking.lost'])

    def test_disabling_tracking_releases_drag(self):
        tracker = self.tracker()
        tracker.gesture_states.update(drag_active=True, pinch_held=True)
        with contextlib.redirect_stdout(io.StringIO()):
            tracker._handle_left_hand_controls(self.hand(True, side='Left'))
        self.assertFalse(tracker.cursor_enabled)
        tracker.system_controller.mouse_up.assert_called_once_with()

    def test_failed_release_retains_state_for_retry(self):
        tracker = self.tracker()
        tracker.gesture_states.update(drag_active=True, pinch_held=True)
        tracker.system_controller.mouse_up.side_effect = [False, True]
        tracker._cancel_pinch()
        self.assertTrue(tracker.gesture_states['drag_active'])
        tracker._cancel_pinch()
        self.assertFalse(tracker.gesture_states['drag_active'])

    def test_capture_exit_reset_and_keyboard_disable_release_drag(self):
        for keys in ([27], [ord('r'), 27], [ord(' '), 27]):
            with self.subTest(keys=keys):
                tracker = self.tracker()
                tracker.sensitivity = .4
                tracker.gesture_states.update(drag_active=True, pinch_held=True)
                tracker.process_frame_optimized = Mock()
                cv2 = Mock()
                cv2.VideoCapture.return_value.read.return_value = (True, object())
                cv2.waitKey.side_effect = keys
                with patch.dict(namespace, cv2=cv2, OptimizedEMASmoothing=Mock()), contextlib.redirect_stdout(io.StringIO()):
                    tracker.run_optimized()
                tracker.system_controller.mouse_up.assert_called_once_with()
                tracker.system_controller.left_click.assert_not_called()

    def test_reset_preserves_failed_release_until_retry(self):
        tracker = self.tracker()
        tracker.sensitivity = .4
        tracker.gesture_states.update(drag_active=True, pinch_held=True, pinch_start_time=10)
        tracker.system_controller.mouse_up.side_effect = [False, True]
        seen = []
        tracker.process_frame_optimized = Mock(side_effect=lambda frame: seen.append(dict(tracker.gesture_states)))
        cv2 = Mock()
        cv2.VideoCapture.return_value.read.return_value = (True, object())
        cv2.waitKey.side_effect = [ord('r'), ord('r'), 27]
        with patch.dict(namespace, cv2=cv2, OptimizedEMASmoothing=Mock()), contextlib.redirect_stdout(io.StringIO()):
            tracker.run_optimized()
        self.assertTrue(seen[1]['drag_active'])
        self.assertTrue(seen[1]['pinch_held'])
        self.assertEqual(seen[1]['pinch_start_time'], 10)
        self.assertFalse(seen[2]['drag_active'])
        self.assertEqual(tracker.system_controller.mouse_up.call_count, 2)
        tracker.system_controller.left_click.assert_not_called()

    def test_right_hand_release_retries_without_click(self):
        tracker = self.tracker()
        tracker._move_cursor_optimized = Mock()
        tracker.gesture_states.update(drag_active=True, pinch_held=True, pinch_start_time=10)
        tracker.system_controller.mouse_up.side_effect = [False, True]
        with contextlib.redirect_stdout(io.StringIO()), patch.object(time, 'time', return_value=11):
            tracker._handle_right_hand_cursor(self.hand(False), 640, 480)
            self.assertTrue(tracker.gesture_states['pinch_held'])
            self.assertTrue(tracker.gesture_states['drag_active'])
            self.assertEqual(tracker.gesture_states['pinch_start_time'], 10)
            tracker._handle_right_hand_cursor(self.hand(False), 640, 480)
        self.assertFalse(tracker.gesture_states['pinch_held'])
        self.assertFalse(tracker.gesture_states['drag_active'])
        self.assertEqual(tracker.system_controller.mouse_up.call_count, 2)
        tracker.system_controller.left_click.assert_not_called()

    def test_quick_pinch_clicks_once_only_in_normal_mode(self):
        for events_only in (False, True):
            tracker = self.tracker(events_only)
            tracker._move_cursor_optimized = Mock()
            with contextlib.redirect_stdout(io.StringIO()), patch.object(time, 'time', return_value=10):
                tracker._handle_right_hand_cursor(self.hand(True), 640, 480)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(time, 'time', return_value=10.1):
                tracker._handle_right_hand_cursor(self.hand(False), 640, 480)
                tracker._handle_right_hand_cursor(self.hand(False), 640, 480)
            self.assertEqual(tracker.system_controller.left_click.call_count, 0 if events_only else 1)

    def test_entrypoints_remain_identical(self):
        self.assertEqual((ROOT / 'main.py').read_bytes(), (ROOT / 'main_ultra_fast_optimized.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
