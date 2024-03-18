"""
Reading a video file using PyTorch
"""

import torchvision
from torchvision.io import read_video

# Path to your MS ASL video file
video_path = '/Users/faridkarimli/Desktop/Programming/AI/Computer Vision/VideoASL/MS-ASL/sample_videos/BOOK.mp4'

# Load the video
video, audio, info = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')

print("Video shape:", video.shape)
print(video)

# `video` is a tensor of shape (T, H, W, C) where T is the number of frames
# H and W are the height and width of the video, and C is the number of channels (usually 3 for RGB).
# `audio` is a tensor for the audio data (we don't need this)
# `info` contains metadata about the video.

