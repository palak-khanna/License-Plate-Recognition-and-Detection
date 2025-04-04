import streamlit as st
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from PIL import Image
import easyocr
from ultralytics import YOLO
import tempfile
import os
import pandas as pd

# Load YOLO models
vehicle_model = YOLO('vehicle_detection_model.pt')
plate_model = YOLO('license_plate_detection_model.pt')

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Database setup
def setup_database():
    conn = sqlite3.connect('license_plates.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS plates (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      plate_number TEXT UNIQUE,
                      vehicle_type TEXT,
                      timestamp TEXT)''')
    conn.commit()
    conn.close()

setup_database()

def detect_vehicle_and_plate(frame):
    """Detects vehicles, extracts license plates, and recognizes text."""
    vehicle_detected, plate_detected = False, False
    class_v, license_plate_text = "NONE", ""
    vehicle_crop, license_plate_crop = None, None

    # Convert frame to OpenCV format
    image_cv = np.array(frame)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Vehicle detection
    vehicle_results = vehicle_model.predict(image_cv)
    names = ['Auto-Rickshaw', 'Bike', 'Bus', 'Car', 'HCV', 'LCV', 'Toto']

    for result in vehicle_results:
        for box in result.boxes:
            vehicle_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_v = names[class_id]
            vehicle_crop = image_cv[y1:y2, x1:x2]
            break

    if vehicle_detected:
        plate_results = plate_model.predict(vehicle_crop)
        for result in plate_results:
            for box in result.boxes:
                plate_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                license_plate_crop = vehicle_crop[y1:y2, x1:x2]
                break

    if plate_detected:
        gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray_plate)
        license_plate_text = "".join([text for (_, text, _) in results]).strip()

    return class_v, vehicle_crop, license_plate_crop, license_plate_text

def save_to_database(plate_number, vehicle_type):
    """Save results to the database, ensuring unique entries."""
    conn = sqlite3.connect('license_plates.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Only insert if the plate number is not already present
    cursor.execute('INSERT OR IGNORE INTO plates (plate_number, vehicle_type, timestamp) VALUES (?, ?, ?)',
                   (plate_number, vehicle_type, timestamp))
    conn.commit()
    conn.close()

def process_video(video_path):
    """Extracts frames from video & processes vehicle detection."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps  # Process 1 frame per second
    current_frame = 0
    detected_vehicles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            class_v, vehicle_crop, license_plate_crop, license_plate_text = detect_vehicle_and_plate(frame_pil)

            if license_plate_text and license_plate_text not in detected_vehicles:
                detected_vehicles.append(license_plate_text)
                save_to_database(license_plate_text, class_v)

                st.subheader(f"Detected Vehicle Type: {class_v}")
                st.image(vehicle_crop, caption="Cropped Vehicle", use_container_width=True)
                st.subheader(f"License Plate Number: {license_plate_text}")
                st.image(license_plate_crop, caption="Cropped License Plate", use_container_width=True)
                st.success("Saved to database!")

        current_frame += 1

    cap.release()
    return detected_vehicles

def main():
    st.title("🎥 License Plate Recognition from Video")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("✅ Video uploaded successfully!")
        st.write("🚀 Processing video...")

        detected_vehicles = process_video(video_path)

        if detected_vehicles:
            st.success(f"✅ Detected {len(detected_vehicles)} unique license plates.")

    if st.button("View Database Records"):
        conn = sqlite3.connect('license_plates.db')
        df = pd.read_sql_query("SELECT * FROM plates", conn)
        st.dataframe(df)
        conn.close()

if __name__ == "__main__":
    main()
