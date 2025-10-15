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
model = YOLO('/home/carbucks/ml/car_parts/runs/detect/Carparts train/weights/best.pt')


# Make predictions on an image
results = model.predict(
    source = '/home/carbucks/Carbucks_YOLO/images/test/003956.jpg',            
    imgsz = 640,
    device = '0',
    )

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