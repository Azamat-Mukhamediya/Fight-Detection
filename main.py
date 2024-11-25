from ultralytics import YOLO
import sys
import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import random
import tensorflow as tf

from ultralytics.utils.plotting import Annotator
from collections import defaultdict


def crop_and_pad(frame, box, margin_percent):
    """Crops a region from the frame with specified margins and resizes it to a square shape.

    Args:
        frame (np.ndarray): The input image frame from the video.
        box (tuple): Coordinates of the bounding box as (x1, y1, x2, y2).
        margin_percent (float): Percentage margin to add around the bounding box.

    Returns:
        np.ndarray: The cropped and resized square image.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin to the bounding box
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Calculate size of square crop and take it from frame
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
        max(0, center_y - half_size): min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size): min(frame.shape[1], center_x + half_size),
    ]
    resized_crop = cv2.resize(square_crop, (64, 64), interpolation=cv2.INTER_LINEAR)

    return resized_crop


def norm_crop(crop):
    """Normalizes the cropped image by scaling pixel values.

    Args:
        crop (np.ndarray): The cropped image.

    Returns:
        np.ndarray: The normalized cropped image.
    """
    norm_crop = crop / 255
    return norm_crop


def main():
    """Main function to perform object detection and classification on video frames."""

    # Load the classifier model
    classifier = tf.keras.models.load_model('vgg_j.model.h5')

    # Define video source and YOLO model for person tracking
    source = "rtsp:/admin:@10.1.198.6:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
    yolo_model = YOLO("yolov8s.pt")

    # Initialize video capture and output settings
    cap = cv2.VideoCapture(source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = 'test.mp4'
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize tracking and frame processing parameters
    track_history = defaultdict(list)
    frame_counter = 0
    track_ids_to_infer = []
    crops_to_infer = []
    pred_labels = []
    pred_confs = []
    classes = ['Non-Fight', 'Fight']
    skip_frame = 5
    crop_margin_percentage = 1
    num_video_sequence_samples = 10
    video_cls_overlap_ratio = 0.25

    # Main loop for reading frames and processing objects
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1

        # Run YOLO tracking on the frame, targeting the person class
        results = yolo_model.track(frame, persist=True, classes=[0])  # Track only person class

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # Initialize the annotator for visualizing predictions
            annotator = Annotator(frame, line_width=3, font_size=10, pil=False)

            if frame_counter % skip_frame == 0:
                crops_to_infer = []
                track_ids_to_infer = []

            for box, track_id in zip(boxes, track_ids):
                if frame_counter % skip_frame == 0:
                    crop_ = crop_and_pad(frame, box, crop_margin_percentage)
                    crop = norm_crop(crop_)
                    track_history[track_id].append(crop)

                # Ensure tracking history has a maximum length
                if len(track_history[track_id]) > num_video_sequence_samples:
                    track_history[track_id].pop(0)

                # Append crops to infer if sequence samples are ready
                if len(track_history[track_id]) == num_video_sequence_samples and frame_counter % skip_frame == 0:
                    crops_to_infer.append(np.asarray(track_history[track_id]))
                    track_ids_to_infer.append(track_id)

            # Run inference if there are crops to infer or overlap condition is met
            if crops_to_infer and (
                    not pred_labels or frame_counter %
                    int(num_video_sequence_samples * skip_frame * (1 - video_cls_overlap_ratio)) == 0):

                pred_labels, pred_confs = [], []

                start_inference_time = time.time()
                for frames in crops_to_infer:
                    frames = np.expand_dims(frames, axis=0)
                    fr = np.expand_dims(frames, axis=-1)
                    pred_proba = classifier.predict(fr)[0]
                    pred_label = 1 if pred_proba >= 0.5 else 0
                    pred_class = classes[pred_label]

                    pred_labels.append(pred_class)
                    pred_confs.append(pred_proba[0])

                end_inference_time = time.time()
                inference_time = end_inference_time - start_inference_time
                print(f"video cls inference time: {inference_time:.4f} seconds")

            # Annotate frame with bounding boxes and labels
            if track_ids_to_infer:
                for box, track_id in zip(boxes, track_ids_to_infer):
                    annotator.box_label(box, 'Person', color=(0, 0, 255))

            if track_ids_to_infer and crops_to_infer:
                for box, track_id, pred_label, pred_conf in zip(
                        boxes, track_ids_to_infer, pred_labels, pred_confs):
                    label_text = f"{pred_label} ({pred_conf:.2f})"
                    annotator.box_label(box, label_text, color=(0, 0, 255))

        # Write the annotated frame to the output video
        if output_path is not None:
            out.write(frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking with S3D Classification", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
