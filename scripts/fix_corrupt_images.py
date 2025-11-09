from ml_carbucks.utils.data_corrections import fix_corrupt_images


if __name__ == "__main__":
    img_dir = "/home/bachelor/ml-carbucks/datasets/carbucks_dataset_v2/images"
    fix_corrupt_images(img_dir)
