"""
Download the videos from the MSASL dataset
"""
from pytube import YouTube
import os
import sys
import json
import requests


cwd = os.getcwd()

DATASET_PATH = f""

splits = ["train", "val", "test"]
paths = [f"{cwd}/MSASL_train.json", f"{cwd}/MSASL_val.json", f"{cwd}/MSASL_test.json"]
print(paths)


def progress_function(stream, chunk, bytes_remaining):
    print(bytes_remaining)


def complete_function(stream, file_handle):
    print("Download complete")


def download_video(video_url, output_path, file_name):
    try:
        yt = YouTube(video_url, on_progress_callback=progress_function, on_complete_callback=complete_function)
        yt.streams.filter(progressive=True, file_extension="mp4").first().download(output_path=output_path,
                                                                                   filename=file_name)
    except Exception as e:
        print(e)


class_file = json.load(open("MSASL_classes.json", "r"))

for (split, path) in zip(splits, paths):
    with open(path, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        url = data[i]["url"]
        label = data[i]["label"]
        class_name = class_file[label]
        print("Downloading video {} with label {}".format(url, label))
        download_video(url, output_path=f"{DATASET_PATH}/{split}/{class_name}",
                       file_name=f"{label}_{data[i]['clean_text']}.mp4")
