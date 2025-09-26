from ultralytics import YOLO

import torch

torch.cuda.empty_cache() 

# Dataset config
data_config_path = "Carbucks_YOLO/dataset.yaml"
num_epochs = 80              # 20 is too low; go 80–150 for better convergence
image_size = 1024            # Larger input boosts accuracy (esp. for small objects)

# Batch size tuning (auto fits GPU if you pass -1, but 64–128 usually fits in 20GB)
batch_size = -1

# Training setup
gpu_device = 0
print("Starting YOLOv8 training...")
print(f"Dataset: {data_config_path}")
print(f"Epochs: {num_epochs}")
print(f"Image Size: {image_size}")
print(f"Batch Size: {batch_size}")


model = YOLO("yolov8l.pt")

results = model.train(
    data=data_config_path,
    epochs=100,
    imgsz=1024,
    batch=8,
    device=gpu_device,
    workers=8,
    amp=True,
    optimizer="AdamW",      # safer for large models
    lr0=2e-4,               # initial learning rate
    lrf=0.01,               # final LR fraction (cosine decay)
    name="yolov8l_overnight_tuning"
)
print("Training finished.")
print("Results saved to the 'runs/detect/yolov8l_best_training' directory.")
