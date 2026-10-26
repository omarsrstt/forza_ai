import cv2
import time
import json
import pygame
import keyboard
import numpy as np
from PIL import Image
from mss import mss
from gamepad import GamepadWriter
from drive_cache import FrameWindow

import torch
import preprocess
import train_model

# Training and inference must agree on frames per second and on the temporal
# window, so both are fixed here rather than read from a second config copy.
FPS_CAP = 15


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
    clock = pygame.time.Clock()

    # Mirror the training pipeline exactly: RGB, INTER_AREA resize, no-jitter
    # ToTensor + ImageNet Normalize. The normalisation constants come from
    # preprocess so training and inference cannot drift apart.
    transform = preprocess.build_train_transform(color_jitter=False)

    # Load the model
    device = "cuda"
    model, _ = train_model.load_forza_checkpoint(config["MODEL_PATH"])
    model.eval().to(device)

    # Rolling window of per-frame CNN features; replaces any hidden state.
    frame_window = FrameWindow(k=config["sequence_length"])

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

                # Grab the screen; mss returns BGRA, training JPEGs decode as RGB.
                frame = np.array(sct.grab(config["MONITOR_REGION"]))
                rgb = preprocess.rgb_from_bgra(frame)
                rgb = cv2.resize(rgb, tuple(config["RESIZE_DIMS"]), interpolation=cv2.INTER_AREA)
                x = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

                # Encode only the newest frame, then run the LSTM over the window.
                feature = model.encode(x).squeeze(0)
                window = frame_window.push(feature)
                outputs = model.forward_from_features(window)

                # Physical outputs already: steer in [-1,1], throttle/brake in [0,1].
                steer, throttle, brake = outputs[0, -1]
                gamepad_writer.update(float(steer), float(throttle), float(brake))

                # Recorded frames were captured at 15 FPS, so drive at 15 FPS too.
                clock.tick(FPS_CAP)
