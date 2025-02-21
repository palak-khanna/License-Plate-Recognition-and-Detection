!pip install ultralytics

import os

train_path = "/content/drive/MyDrive/License plate detection and recognition/train/images"
val_path = "/content/drive/MyDrive/License plate detection and recognition/valid/images"

print("Train path exists:", os.path.exists(train_path))
print("Validation path exists:", os.path.exists(val_path))


import os

base_path = "/content/drive/MyDrive/License plate detection and recognition"
paths = [
    os.path.join(base_path, "train/images"),
    os.path.join(base_path, "valid/images"),
    os.path.join(base_path, "test/images")
]

for path in paths:
    print(f"Path: {path}, Exists: {os.path.exists(path)}")


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for Nano, or switch to 'yolov8s.pt' for Small, etc.

# Train the model
model.train(
    data="/content/drive/MyDrive/License plate detection and recognition/data.yaml",  # Path to the dataset configuration file
    epochs=5,                 # Number of epochs
    imgsz=640,                 # Image size
    batch=16                   # Batch size (adjust based on GPU memory)
)

import os
source = '/content/drive/MyDrive/License plate detection and recognition/test/images'
if os.path.exists(source):
    print("Path exists")
else:
    print("Path does not exist")

from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference on the test dataset
results = model.predict(source='/content/drive/MyDrive/License plate detection and recognition/test/images', save=True, conf=0.5)

# Display results
from IPython.display import Image, display
import glob

# Show saved images with predictions
image_files = glob.glob('runs/detect/predict/*.jpg')  # Path where YOLO saves predictions
for image_file in image_files:
    display(Image(filename=image_file))

from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Perform validation
metrics = model.val()

# Print metrics
print("Validation Metrics:")
print(f"mAP@0.5: {metrics.box.map50:.2f}")       # Mean Average Precision at IoU 0.5
print(f"mAP@0.5:0.95: {metrics.box.map:.2f}")   # Mean Average Precision at IoU 0.5 to 0.95
print(f"Precision: {metrics.box.mp:.2f}")       # Mean Precision
print(f"Recall: {metrics.box.mr:.2f}")          # Mean Recall

# Display the saved validation results
import glob
from IPython.display import Image, display

val_images = glob.glob('runs/detect/val/*.jpg')  # Path where validation predictions are saved
for val_image in val_images:
    display(Image(filename=val_image))

import glob
from IPython.display import Image, display

# Path to validation results
val_dir = 'runs/detect/val/'

# Display evaluation visualizations (precision-recall curves, etc.)
print("Displaying Accuracy Visualizations:")

# Show precision-recall curves
pr_curve = glob.glob(val_dir + 'PR_curve.png')
if pr_curve:
    display(Image(filename=pr_curve[0]))
else:
    print("Precision-Recall curve not found.")

# Show confusion matrix (if available)
confusion_matrix = glob.glob(val_dir + 'confusion_matrix.png')
if confusion_matrix:
    display(Image(filename=confusion_matrix[0]))
else:
    print("Confusion matrix not found.")

# Additional images in the val directory
additional_images = glob.glob(val_dir + '*.png')
for img in additional_images:
    if img not in pr_curve and img not in confusion_matrix:
        display(Image(filename=img))

# Save the trained model
model.save('license_plate_detection_model.pt')

from google.colab import files
files.download('license_plate_detection_model.pt')

