from ml_carbucks.utils.conversions import convert_yolo_to_coco


if __name__ == "__main__":
    base_directory = "/home/bachelor/ml-carbucks/datasets/carbucks_dataset_v2"
    dataset_splits = ["train", "val", "test"]

    convert_yolo_to_coco(base_directory, dataset_splits)
