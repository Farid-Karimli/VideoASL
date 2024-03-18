"""
Annotating videos with the bounding box
"""

import os
import json
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

train_path = "MSASL_train.json" # Path to the training data, assuming you have it in the current directory

with open(train_path, "r") as f:
    data = json.load(f)

# BOOK sign
sample = data[3]
WIDTH, HEIGHT = int(sample["width"]), int(sample["height"])
print(sample)

# Load the video
video_path = "./sample_videos/BOOK.mp4"
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video length:", length)

bounding_box = sample["box"]
bounding_box[0], bounding_box[1] = int(bounding_box[0]*sample["width"]), int(bounding_box[1]*sample["height"])
bounding_box[2], bounding_box[3] = int(bounding_box[2]*sample["width"]), int(bounding_box[3]*sample["height"])

print("Bounding box:", bounding_box)

# Read the video frame by frame
frames = []
for i in range(length):
    ret, frame = cap.read()
    if ret:
        # Draw the bounding box
        x, y, x1, y1 = bounding_box
        frame = cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        frames.append(frame)
    else:
        break

cap.release()

# save the video
os.makedirs("annotated_videos", exist_ok=True)
out = cv2.VideoWriter("annotated_videos/BOOK.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(sample['width']),
                                                                                         int(sample['height'])))
for frame in frames:
    out.write(frame)
out.release()


