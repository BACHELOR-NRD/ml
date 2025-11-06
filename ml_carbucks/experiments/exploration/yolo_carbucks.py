from ultralytics.models.yolo import YOLO

model = YOLO("yolo11l.pt")

data_yaml = "/home/bachelor/ml-carbucks/data/carbucks/dataset.yaml"

model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo_carbucks_experiment",
    project="/home/bachelor/ml-carbucks/results/debug/carbucks_ultralytics",
)
