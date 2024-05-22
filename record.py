import os
import cv2
import time
import json
import pygame
import keyboard
import numpy as np
from mss import mss
from gamepad import GamepadListener, GamepadWriter

def start_buffer():
    ### wait some time
    for i in reversed(list(range(5))):
        print(i)
        time.sleep(0.5)

# Region of screen capture
# monitor_region = {"top": 40, "left": -1920, "width": 1730, "height": 960}
# monitor_region = {"top": 40, "left": -1920, "width": 970, "height": 550}
# monitor_region = {"top": 0, "left": -1920, "width": 1920, "height": 1080}
monitor_region = {"top": 0, "left": -1920, "width": 1920, "height": 1080}


def main():
    # Timing
    last_time = time.perf_counter()
    clock = pygame.time.Clock()

    # Initialize gamepad listener
    gamepadlistener = GamepadListener()

    # Resize input image
    resize_dims = (224, 224)

    # Create folder to store recorded data
    output_dir = "dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get sequence prefix for storing the data
    existing_files = os.listdir(output_dir)
    sequence_numbers = [int(file_name.split('_')[1]) for file_name in existing_files if file_name.startswith('seq') and file_name.split('_')[1].isdigit()]
    max_seq_num = 0 if not sequence_numbers else max(sequence_numbers)
    sequence_number = max(max_seq_num, 0) + 1
    prefix = "seq_" + str(sequence_number)

    # File index for saves
    index = 1

    start_buffer()

    with mss() as sct:
        while True:
            # Capture the screen
            image = np.array(sct.grab(monitor_region))
            resized_image = cv2.resize(image, resize_dims, interpolation=cv2.INTER_AREA)

            # Get controller events
            events = gamepadlistener.get_joystick_events()

            # Save images
            image_path = os.path.join(output_dir, f"{prefix}_image_{sequence_number}_{index}.jpg")
            cv2.imwrite(image_path, resized_image)
            
            # Save events
            events_path = os.path.join(output_dir, f"{prefix}_events_{sequence_number}_{index}.json")
            with open(events_path, 'w') as f:
                json.dump(events.tolist(), f)


            # Note execution time
            current_time = time.perf_counter()
            print(f"Time elapsed :{current_time - last_time}")
            last_time = current_time

            # Limit recording to 30 FPS
            clock.tick(30)

            # Increment index
            index += 1

            # Exit loop if 'q' is pressed
            if keyboard.is_pressed('q'):
                break
    
    # Close OpenCV windows
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()