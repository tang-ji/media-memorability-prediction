import os
import sys
import pandas as pd
import cv2

def store_frames(videos, sources_path, frames_path):
    """ Extract and store frames from videos """
    for video_name in videos:
        
        # Filename
        video_path = os.path.join(sources_path, video_name)
        # Short name
        video_name_short = video_name.split('.webm')[0]

        if not os.path.isfile(video_path):
            print('File {} does not exist'.format(video_path))
            continue

        print('Reading file: {}'.format(video_path))

        # Read Video!
        success = True
        video = cv2.VideoCapture(video_path)
        success, image = video.read()

        # Frame
        frame_id = int(video.get(1))
        # cv2.imwrite("{}/{}-frame-{:03}.jpg".format(frames_path, video_name_short, frame_id), image)
        cv2.imwrite("{}/{}-frame-{}.jpg".format(frames_path, video_name_short, frame_id), image)
        
        # Frames per second
        fps = int(video.get(cv2.CAP_PROP_FPS))
        
        # Collect 1 frame every 1 second
        seconds = 1
        multiplier = fps * seconds

        while success:

            # Frame
            frame_id = int(video.get(1))

            if frame_id % multiplier == 0:
                # Save it
                cv2.imwrite("{}/{}-frame-{}.jpg".format(frames_path, video_name_short, frame_id), image)

            # Read Video!
            success, image = video.read()

        video.release()
        
