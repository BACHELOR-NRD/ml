from ultralytics import YOLO
import os


DATA_YAML = "/home/carbucks/Carbucks_YOLO/dataset.yaml"
PROJECT_DIR = "/home/carbucks/runs/detect"
RUN_NAME = "y11x_run"
PRETRAINED = "yolo11x.pt"
TEST_IMAGES = "/home/carbucks/Carbucks_YOLO/images/test"


os.makedirs(PROJECT_DIR, exist_ok=True)

model = YOLO(PRETRAINED)

# Train with full hyperparameters (edit as needed)
results = model.train(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name=RUN_NAME,
    epochs=100,
    imgsz=1024,
    batch=-1,
    device=0,
    workers=8,
    seed=42,
    deterministic=True,
    optimizer="AdamW",
    pretrained=True,
    save=True,
    save_period=10,
    cache=False,
    val=True,
    amp=False,
    close_mosaic=10,
    rect=False,
    multi_scale=True,
    cos_lr=False,
    lr0=0.001,
    plots=True,
    auto_augment="autoaugment",
    patience=10,
)

# Resolve run paths
run_dir = os.path.join(PROJECT_DIR, RUN_NAME)
weights_d = os.path.join(run_dir, "weights")
best_pt = os.path.join(weights_d, "best.pt")
last_pt = os.path.join(weights_d, "last.pt")

print("\n=== TRAIN OUTPUTS ===")
print(f"Run directory : {run_dir}")
print(f"Best weights  : {best_pt}")
print(f"Last weights  : {last_pt}")


# Reload best model checkpoint
model = YOLO(best_pt)

# Run validation
val_res = model.val(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name=f"{RUN_NAME}_val",
    imgsz=1024,
    batch=0.9,
    device=0,
    save_json=True,
    plots=True,
    augment=True,
    visualize=True,
)

val_dir = os.path.join(PROJECT_DIR, f"{RUN_NAME}_val")
print("\n=== VAL OUTPUTS ===")
print(f"Val directory : {val_dir}")
print(f"mAP50-95      : {val_res.results_dict.get('metrics/mAP50-95(B)', 'n/a')}")
