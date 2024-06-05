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
# MONITOR_REGION = {"top": 40, "left": -1920, "width": 1730, "height": 960}
# MONITOR_REGION = {"top": 40, "left": -1920, "width": 970, "height": 550}
# MONITOR_REGION = {"top": 0, "left": -1920, "width": 1920, "height": 1080}
# MONITOR_REGION = {"top": 0, "left": -1920, "width": 1920, "height": 1080} # Left monitor
MONITOR_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080} # Right monitor


def main():
    # Timing
    last_time = time.perf_counter()
    clock = pygame.time.Clock()

    # Initialize gamepad listener
    gamepadlistener = GamepadListener()

    # Resize input image
    RESIZE_DIMS = (224, 224)

    # Specify record FPS
    RECORD_FPS_CAP = 15

    # Create folder to store recorded data
    OUTPUT_DIR = "dataset"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Get sequence prefix for storing the data
    existing_files = os.listdir(OUTPUT_DIR)
    sequence_numbers = [int(file_name.split('_')[1]) for file_name in existing_files if file_name.startswith('seq') and file_name.split('_')[1].isdigit()]
    
    if not sequence_numbers:
        sequence_number = 0
    else:
        sequence_numbers.sort()
        missing_number = None
        for idx, seq_num in enumerate(sequence_numbers):
            if seq_num != idx:
                missing_number = idx
                break
        sequence_number = missing_number if missing_number else max(sequence_numbers) + 1
    
    prefix = "seq_" + str(sequence_number)

    # File index for saves
    index = 1

    # Wait some time before starting the recording
    start_buffer()

    with mss() as sct:
        while True:
            # Capture the screen
            image = np.array(sct.grab(MONITOR_REGION))
            resized_image = cv2.resize(image, RESIZE_DIMS, interpolation=cv2.INTER_AREA)

            # Get controller events
            events = gamepadlistener.get_joystick_events()

            # Save images
            image_path = os.path.join(OUTPUT_DIR, f"{prefix}_image_{sequence_number}_{index}.jpg")
            cv2.imwrite(image_path, resized_image)
            
            # Save events
            events_path = os.path.join(OUTPUT_DIR, f"{prefix}_events_{sequence_number}_{index}.json")
            with open(events_path, 'w') as f:
                json.dump(events.tolist(), f)


            # Note execution time
            current_time = time.perf_counter()
            print(f"Time elapsed :{current_time - last_time}")
            last_time = current_time

            # Limit recording to 30 FPS
            clock.tick(RECORD_FPS_CAP)

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