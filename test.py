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

def generate_random_events():
    # Activate/Deactivate buttons
    buttons = np.random.choice([0, 1], size=4)
    # L-R Joystick
    lr_joy = (np.random.random(size=1)-0.5)*2 # Generate float [-1, 1]
    # Triggers
    triggers = np.random.rand(2)
    return np.concatenate((buttons, lr_joy, triggers))

def test_gamepad_writer(gamepad_writer):
    events = generate_random_events()
    gamepad_writer.update(events=events)

if __name__ == "__main__":
    gamepad_writer = GamepadWriter()
    while True:
        test_gamepad_writer(gamepad_writer)
        sleep(0.5)