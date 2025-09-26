import os
from ultralytics import YOLO
import cv2

# Set ultralytics config directory
os.environ['ULTRALYTICS_CONFIG_DIR'] = '/home/carbucks/.ultralytics'
os.makedirs(
    '/home/carbucks/.ultralytics', exist_ok=True)

# Create output directory
output_dir = '/home/carbucks/ml/test_run/predictions'
os.makedirs(output_dir, exist_ok=True)

# Load your trained model
model = YOLO('/home/carbucks/ml/test_run/yolov8l_overnight_tuning4/weights/best.pt')


# Make predictions on an image
results = model.predict(
    source = '/home/carbucks/0a21ce16-758d-4d99-a958-bc7f9239a8ad.jpg',            
    imgsz = 1024,
    device = '0',
    visualize = True)

# Process results
for idx, result in enumerate(results):
    if result.boxes is not None:
        boxes = result.boxes.xyxy  # bounding boxes
        conf = result.boxes.conf   # confidence scores
        cls = result.boxes.cls     # class labels
        
        print(f"Found {len(boxes)} detections")
        
        # Draw results on image
        annotated_img = result.plot()
        output_path = os.path.join(output_dir, f"prediction_{idx}.jpg")
        cv2.imwrite(output_path, annotated_img)
        print(f"Saved annotated image to {output_path}")
    else:
        print("No detections found")