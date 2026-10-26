import keyboard
import numpy as np
from time import sleep
from gamepad import GamepadListener, GamepadWriter

# gamepad = GamepadListener()
# if __name__ == "__main__":
#     while True:
#         print(gamepad.get_joystick_events())
#         if keyboard.is_pressed('q'):
#                 break

def generate_random_controls():
    # L-R Joystick (steer)
    steer = (np.random.random() - 0.5) * 2 # Generate float [-1, 1]
    # Throttle / brake
    throttle, brake = np.random.rand(2) # Generate floats [0, 1]
    return steer, throttle, brake

def test_gamepad_writer(gamepad_writer):
    steer, throttle, brake = generate_random_controls()
    gamepad_writer.update(steer=steer, throttle=throttle, brake=brake)

if __name__ == "__main__":
    gamepad_writer = GamepadWriter()
    while True:
        test_gamepad_writer(gamepad_writer)
        sleep(0.5)