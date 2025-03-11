## Forza Automation
### Overview
This is a project for automating driving in Forza Horizon 4 Game. It records input images and preferred driving commands using `record.py`, then trains a model on this data using various libraries and tools. Trained models can then be used to drive the car within the game using `drive.py`


### Creating a dataset
- To train a model, we first have to create a dataset with the input images and preferred driving commands. To do so we have to use the `record.py`
- The data is recorded using a `controller`, so keyboard inputs aren't supported yet.
- The `MONITOR_REGION` variable stores the location of the screen, current it is set to record the left screen in a dual screen setup, this can be changed as needed
- The `OUTPUT_DIR` variable defines the location where the recorded data is stored
- The `RECORD_FPS_CAP` variable defines the frame rate at which the data is recorded
- `RESIZE_DIMS` defines the size of the recorded images/frames
  - Run the record script `python record.py` to start recording the game's output.
  - Press `q` to stop the recording process.
  - Each run is stored as a sequence.

### Training a model
- Once you have recorded data, you can use various models (e.g., `forza-convnext-lstm`) to train and optimize performance.
- The training process uses the PyTorch Lightning framework for easy experiment tracking and logging.
- You can customize hyperparameters in the `config.json` file, such as batch size, sequence length, and learning rate.

### Training Settings
- We use a combination of ConvNeXt Tiny and LSTM models to predict driving commands from game images.
- The training process involves the following steps:
  1. Data loading: Load recorded data sequences into memory using `ForzaDataset` or `ForzaLSTMDataset`.
  2. Model initialization: Initialize the model with either ConvNeXt Tiny or ConvNext Tiny LSTM architecture, depending on the `config.json` settings.
  3. Training loop: Train the model on the loaded data for a specified number of epochs using PyTorch Lightning's `Trainer`.


### Project Structure
- `README.md` - Description of the project/repository
- `drive.py` - Program to drive the car in forza
- `gamepad.py` - Implements classes and relevant methods to record inputs from the controller and to write commands as the controller
- `record.py` - Code to record and create a sequence for the forza dataset
- `train_model.py` - Model trainer script