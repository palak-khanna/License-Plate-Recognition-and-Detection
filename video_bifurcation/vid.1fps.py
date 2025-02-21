import os
import argparse
import cv2

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract 1 frame per second from a video.")
    parser.add_argument('source', type=str, help='Path to source video')
    parser.add_argument('dest_folder', type=str, help='Path to destination folder')
    args = parser.parse_args()

    # Open video file
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Ensure the destination folder exists
    os.makedirs(args.dest_folder, exist_ok=True)

    # Get frames per second (FPS) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Convert to integer
    print(f"Video FPS: {fps}")

    current_frame = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()

        # Break the loop if video ends
        if not ret:
            break

        # Save only 1 frame per second
        if current_frame % fps == 0:  # Save frame every FPS interval
            filename = f"frame_{saved_frame_count:04d}.jpg"
            file_path = os.path.join(args.dest_folder, filename)
            cv2.imwrite(file_path, frame)
            print(f"Saved: {filename}")
            saved_frame_count += 1

        current_frame += 1

    # Release capture 
    cap.release()
    print(f"Extraction complete. {saved_frame_count} frames saved.")

if __name__ == '__main__':
    main()

#            to run do this in cmd
#       python script.py input_video.mp4 output_folder/D:\Ekamjot Kaur\CB\photo from videos
#             "D:\Ekamjot Kaur\CB\vid.1fps.py" "D:\Ekamjot Kaur\CB\videos\5car.webm" "D:\Ekamjot Kaur\CB\photo from videos"
# after a while it restats saving from 00 and re writes over the older photos, so if want to keep them, then make new folder
