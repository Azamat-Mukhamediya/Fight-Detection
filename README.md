# YOLOv8 Person Tracking and Fight Detection

This project uses YOLOv8 to track people in video feeds and a classifier to detect "Fight" vs. "Non-Fight" actions.

## Dependencies

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How It Works

1. **Person Detection**: YOLOv8 identifies people in each frame of the video.
2. **Frame Cropping**: Each detected person is cropped from the frame with a small margin.
3. **Action Classification**: Cropped sequences are passed to a classifier to predict "Fight" or "Non-Fight."
4. **Annotation**: Predictions are displayed with labels on each person.

## Usage

1. **Prepare Models**: Place `yolov8s.pt` and `vgg_j.model.h5` (the classifier model) in the same directory as the script.
2. **Run the Code**:
   ```bash
   python main.py
   ```
3. **Video Source**: Update the `source` variable in the code with your video feed (e.g., RTSP URL).

### Configuration Options

- `output_path`: Path to save the annotated video (default is `test.mp4`).
- `skip_frame`: Number of frames to skip between processing (default is `5`).
- `num_video_sequence_samples`: Number of frames used for classification.

## Output

- **Annotated Video**: The processed video with bounding boxes and action labels is displayed and saved (if `output_path` is set).
- **Real-Time Display**: View live predictions in a display window.

## Example

To use an IP camera feed, set `source` in the code:
```python
source = "rtsp://<your-camera-url>"
```

Press `q` to stop the program.

## Acknowledgments

This project uses the YOLOv8 model provided by [Ultralytics](https://github.com/ultralytics/ultralytics/). We appreciate their efforts in making their code and examples publicly available.

