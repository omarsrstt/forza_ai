import gamepad


class _Rec:
    def __init__(self): self.calls = {}
    def reset(self): pass
    def update(self): pass
    def left_joystick_float(self, x_value_float, y_value_float): self.calls["joy"] = (x_value_float, y_value_float)
    def left_trigger_float(self, value_float): self.calls["lt"] = value_float
    def right_trigger_float(self, value_float): self.calls["rt"] = value_float


def _w():
    w = gamepad.GamepadWriter.__new__(gamepad.GamepadWriter)
    w.gamepad = _Rec(); return w


def test_throttle_goes_to_right_trigger_brake_to_left():
    w = _w(); w.update(steer=0.0, throttle=1.0, brake=0.0)
    assert w.gamepad.calls["rt"] == 1.0 and w.gamepad.calls["lt"] == 0.0


def test_steering_goes_to_left_joystick_x():
    w = _w(); w.update(steer=-0.7, throttle=0.0, brake=0.0)
    assert w.gamepad.calls["joy"] == (-0.7, 0.0)


def test_values_are_clamped_to_valid_range():
    w = _w(); w.update(steer=5.0, throttle=5.0, brake=-5.0)
    assert w.gamepad.calls["joy"][0] == 1.0
    assert w.gamepad.calls["rt"] == 1.0
    assert w.gamepad.calls["lt"] == 0.0
