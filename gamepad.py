import pygame
import logging
import vgamepad as vg
import numpy as np

### For joystick, we are interested in the following buttons ###

# Events
# 1539/1540 -> button press
# 1536 -> controller axes

# Buttons
# 'button': 4 -> L1
# 'button': 5 -> R1
# 'button': 2 -> Square
# 'button': 1 -> Circle

# Triggers
# 'axis': 0, 'value': value in range [-1,1] -> L-R
# 'axis': 5, 'value': value in range [-1,1] -> R2
# 'axis': 4, 'value': value in range [-1,1] -> L2

class GamepadListener():
    def __init__(self, logging_level = logging.INFO):
        self.logging_level = logging_level
        self.codes = ["B_L", "B_R", "B_S", "B_C", "T_LR", "T_R", "T_L"]
        self.previous_triggers = {0: 0., 1: -1., 2: -1.}
        self.events = np.zeros(7)
        self.setup_logging()
        self.setup_joystick()

    def __del__(self):
        pygame.quit()  # Quit Pygame in the destructor

    def set_debug_logging(self):
        self.logging_level = logging.DEBUG
        self.setup_logging()
        logging.debug("Logging level set to DEBUG")

    def setup_logging(self):
        logging.basicConfig(level=self.logging_level)
        logging.debug("Logging setup")

    def setup_joystick(self):
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.debug("Joystick setup")
        # self.clock = pygame.time.Clock()

    def get_joystick_events(self):
        self.events = np.zeros(7)
        for event in pygame.event.get():
            if event.type == 1539 or event.type == 1540: # Button Press
                self.handle_button_press(event.dict['button'])
            elif event.type == 1536: # Joystick movement
                self.handle_axis_movement(event.dict['axis'], event.dict['value'])
        # Update previous trigger values for continuous reading
        self.update_trigger_values()
        return self.events
    
    def handle_button_press(self, button):
        if button == 4:
            self.events[0] = 1
            logging.debug("Button L1 pressed")
        elif button == 5:
            self.events[1] = 1
            logging.debug("Button R1 pressed")
        elif button == 2:
            self.events[2] = 1
            logging.debug("Button Square pressed")
        elif button == 1:
            self.events[3] = 1
            logging.debug("Button Circle pressed")

    def handle_axis_movement(self, axis, value):
        if axis == 0:  # Left-Right joystick
            self.previous_triggers[0] = value
            logging.debug(f"Left-Right joystick used {value}")
        elif axis == 5: # R2 trigger
            self.previous_triggers[1] = value
            logging.debug(f"R2 trigger used {value}")
        elif axis == 4: # L2 trigger
            self.previous_triggers[2] = value
            logging.debug(f"L2 trigger used {value}")

    def update_trigger_values(self):
        self.events[4] = self.previous_triggers[0]
        self.events[5] = self.previous_triggers[1]
        self.events[6] = self.previous_triggers[2]


class GamepadWriter():
    def __init__(self) -> None:
        # Create Xbox 360 gamepad
        self.gamepad = vg.VX360Gamepad()
        self.gamepad.reset()

        # Create dict to store controller commands info
        # [L1, R1, Square, Circle, LR, R2, L2]
        self.events = np.zeros(7)

    def update(self, events):
        self.events = events
        
        ### Button states
        # L1/LB control
        if self.events[0] == 1:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        
        # R1/RB control
        if self.events[1] == 1:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

        # Square/X control
        if self.events[2] == 1: 
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        else: 
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

        # Circle/B control
        if self.events[3] == 1:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)


        # Update LR state
        self.gamepad.left_joystick_float(x_value_float=self.events[4], y_value_float=0.0)

        # Update trigger states
        self.gamepad.left_trigger_float(value_float=self.events[5])
        self.gamepad.right_trigger_float(value_float=self.events[6])

        # Update and send commands
        self.gamepad.update()

    def reset_controller_state(self):
        self.gamepad.reset()


if __name__ == "__main__":
    gamepad_listener = GamepadListener(logging_level=logging.DEBUG)
    while True:
        gamepad_listener.get_joystick_events()