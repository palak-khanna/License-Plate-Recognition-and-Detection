import os
import numpy as np
import argparse
import cv2

def main():

    # Parse all args
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Path to source video')
    parser.add_argument('dest_folder', type=str, help='Path to destination folder')
    args = parser.parse_args()

    # Get file path for desired video and where to save frames locally
    cap = cv2.VideoCapture(args.source)
    path_to_save = os.path.abspath(args.dest_folder)
    
    current_frame = 1
    saved_frame_count = 0
    frame_skip = 24 # Save every 4th frame

    if not cap.isOpened():
        print('Cap is not open')
        return
