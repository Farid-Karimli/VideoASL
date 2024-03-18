"""
Download the videos from the MSASL dataset
"""



from pytube import YouTube
import os
import sys
import json
import requests

# assuming you have installed the MSASL dataset in this directory
PATH = "./MSASL_train.json"


def progress_function(stream, chunk, bytes_remaining):
    print(bytes_remaining)


def complete_function(stream, file_handle):
    print("Download complete")


with open(PATH, "r") as f:
    data = json.load(f)

for i in range(0, 10):
    print(data[i]["url"])
    url = data[i]["url"]
    label = data[i]["label"]

    try:
        yt = YouTube(url, on_progress_callback=progress_function, on_complete_callback=complete_function)
        yt.streams.filter(progressive=True, file_extension="mp4").first().download(output_path="./sample_videos")
    except Exception as e:
        print(e)
        continue
