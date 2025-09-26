from ultralytics import YOLO

import torch

torch.cuda.empty_cache() 

# Dataset config
data_config_path = "/home/carbucks/Carbucks_YOLO/car_parts/config.yaml"
num_epochs = 50            # 20 is too low; go 80–150 for better convergence
image_size = 640            # Larger input boosts accuracy (esp. for small objects)

# Batch size tuning (auto fits GPU if you pass -1, but 64–128 usually fits in 20GB)
batch_size = 0.9

# Training setup
gpu_device = 0
print("Starting YOLOv11 training...")
print(f"Dataset: {data_config_path}")
print(f"Epochs: {num_epochs}")
print(f"Image Size: {image_size}")
print(f"Batch Size: {batch_size}")


model = YOLO("/home/carbucks/ml/car_parts/runs/detect/Carparts train/weights/best.pt")

results = model.train(
    data=data_config_path,
    epochs=num_epochs,
    imgsz=image_size,
    batch=batch_size,
    device=gpu_device,
    workers=8,
    amp=True,
    patience = 5,
    cache = True,
    exist_ok = True,
    seed = 42,
    resume = True,
    optimizer="AdamW",      # safer for large models
    lr0=2e-4,               # initial learning rate
    lrf=0.01,               # final LR fraction (cosine decay)
    name="Carparts train"
)
print("Training finished.")
print("Results saved to the 'runs/detect/yolov8l_best_training' directory.")
