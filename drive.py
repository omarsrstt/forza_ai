import cv2
import time
import json
import keyboard
import numpy as np
from mss import mss
from record import start_buffer
import matplotlib.pyplot as plt
from gamepad import GamepadWriter

import torch
import torch.nn.functional as F
from train_model import ConvNeXtTinyRegression, ConvNextTinyLSTMRegression

def check_pause(paused : bool):
    if keyboard.is_pressed('p'):
        paused = not paused
        if paused:
            print("Paused. Press 'p' to resume.")
        # Wait for a moment to avoid detecting the same key press multiple times
        while keyboard.is_pressed('p'):
            time.sleep(0.1)
    return paused


if __name__ == "__main__":

    # Load the config file
    with open('drive_config.json', 'r') as file:
        config = json.load(file)

    paused = False
    gamepad_writer = GamepadWriter()

    # Load the model
    MODEL_PATH = config["MODEL_PATH"]
    if config["LSTM"]:
        model = ConvNextTinyLSTMRegression()
        model.load_state_dict(torch.load(MODEL_PATH), strict=False)
        model.reset_hidden_state()
    else:
        model = ConvNeXtTinyRegression()
        model.load_state_dict(torch.load(MODEL_PATH), strict=False)

    # model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    model.eval().to("cuda")

    # Add a small time buffer before the start
    # start_buffer()

    with mss() as sct:
        with torch.no_grad():
            while True:

                # Exit loop if 'q' is pressed
                if keyboard.is_pressed('q'):
                    print("Exiting program ...")
                    break

                # Check and handle pause
                paused = check_pause(paused)

                # If paused, wait until resumed
                while paused:
                    paused = check_pause(paused)
                    time.sleep(0.1)  # Prevent busy waiting

                # Get input screen capture
                image = np.array(sct.grab(config["MONITOR_REGION"]))
                # Convert to PyTorch tensor and remove alpha channel
                torch_image = torch.from_numpy(image[:, :, :3]).float().to("cuda")  # Shape: (1080, 1920, 3)

                # Resize and preprocess
                torch_image = torch_image.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW, add batch dimension
                torch_image = F.interpolate(torch_image, size=config["RESIZE_DIMS"], mode="bilinear", align_corners=False)  # Resize
                mean = torch.tensor(config["normalize_mean"], device="cuda").view(1, 3, 1, 1)  # Normalize mean
                std = torch.tensor(config["normalize_std"], device="cuda").view(1, 3, 1, 1)    # Normalize std
                torch_image = (torch_image - mean) / std  # Normalize

                # if config["LSTM"]:
                    # torch_image = torch_image.unsqueeze(0)                        

                # Forward pass
                outputs = model(torch_image)

                # Convert the tensor to a list
                output_list = outputs[0].tolist()

                ### Drive post processing ###
                # Reduce impact of brakes
                output_list[6] = output_list[6]/5 if output_list[6] != 0 else 0
                # Increase impact of steering
                output_list[4] = output_list[4]*10
                # Ensure that the 3 trigger values are in the range [-1, 1]
                output_list[4:] = [max(-1, min(1, num)) for num in output_list[4:]]
                # First 4 elements are button inputs, i.e. boolean values
                output_list[:4] = [int(np.round(x)) for x in output_list[:4]]
                assert all(x in {0, 1} for x in output_list[:4]), f"Got an incorrect value for button press {output_list}"

                print(output_list)

                # Generate controls on this basis
                gamepad_writer.update(output_list)