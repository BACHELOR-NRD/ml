import os
import yaml
from pathlib import Path

# Check dataset structure
DATA_YAML = "/home/carbucks/Carbucks_YOLO/dataset.yaml"

# Load and verify YAML
with open(DATA_YAML, 'r') as f:
    data = yaml.safe_load(f)
    print("YAML contents:", data)

# Check paths
base_path = Path(data['path'])
val_images_path = base_path / data['val']
val_labels_path = base_path / 'labels' / 'val'

print(f"Base path: {base_path}")
print(f"Val images path: {val_images_path}")
print(f"Val labels path: {val_labels_path}")

print(f"Base path exists: {base_path.exists()}")
print(f"Val images path exists: {val_images_path.exists()}")
print(f"Val labels path exists: {val_labels_path.exists()}")

if val_images_path.exists():
    image_files = list(val_images_path.glob('*.jpg')) + list(val_images_path.glob('*.png'))
    print(f"Number of image files: {len(image_files)}")
    print(f"First few images: {[f.name for f in image_files[:5]]}")

if val_labels_path.exists():
    label_files = list(val_labels_path.glob('*.txt'))
    print(f"Number of label files: {len(label_files)}")
    print(f"First few labels: {[f.name for f in label_files[:5]]}")
else:
    print("Labels directory doesn't exist!")

# Test with a simple validation
from ultralytics import YOLO
BEST_PT = "/home/carbucks/runs/detect/y11x_run/weights/best.pt"

try:
    m = YOLO(BEST_PT)
    # Save validation results to the same run directory
    val_res = m.val(
        data=DATA_YAML,
        split='val',
        batch=1,
        device=0,
        verbose=True,
        project="/home/carbucks/runs/detect",
        name="y11x_run/validation",
        save_json=True,
        plots=True
    )
    print("Validation successful!")
    print(f"Results saved to: {val_res.save_dir}")
except Exception as e:
    print(f"Validation error: {e}")
    import traceback
    traceback.print_exc()