## Forza Automation

### Creating a dataset
- To train a model, we first have to create a dataset with the input images and preferred driving commands. To do so we have to use the `record.py`
- The data is recorded using a `controller`, so keyboard inputs aren't supported yet.
- The `MONITOR_REGION` variable stores the location of the screen, current it is set to record the left screen in a dual screen setup, this can be changed as needed
- The `OUTPUT_DIR` variable defines the location where the recorded data is stored
- The `RECORD_FPS_CAP` variable defines the frame rate at which the data is recorded
- `RESIZE_DIMS` defines the size of the recorded images/frames

Initialize the game and then start recording, to stop the recording process, press `q`. Data recorded in a single run is stored as a sequence

### We then train a model on this data