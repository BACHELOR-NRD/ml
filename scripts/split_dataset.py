from pathlib import Path
import random


def split_dataset(
    base_dir: str,
    split_ratio: float = 0.2,
    splits: list = ["train", "val"],
    limit: int = int(1e6),
):
    """
    This function moves all the yolo splits to train first and then moves a percentage
    of the files to val based on the split_ratio.
    """
    files_moved = 0
    files_not_moved = 0

    # move everything to train first
    for split in splits:
        img_dir_path = Path(base_dir) / "images" / split
        for img_path in img_dir_path.glob("*.jpg"):
            if split == "val":
                new_img_dir_path = img_dir_path.parent / "train"
                new_img_dir_path.mkdir(exist_ok=True)
                img_path.rename(new_img_dir_path / img_path.name)

                label_file_path = (
                    img_dir_path.parent.parent
                    / "labels"
                    / split
                    / (img_path.stem + ".txt")
                )
                new_label_dir_path = label_file_path.parent.parent / "train"
                new_label_dir_path.mkdir(exist_ok=True)
                label_file_path.rename(new_label_dir_path / label_file_path.name)

                print(f"Moving {img_path} to {new_img_dir_path / img_path.name}")
                print(
                    f"Moving {label_file_path} to {new_label_dir_path / label_file_path.name}"
                )

    # move to val based on split ratio

    img_dir_path = Path(base_dir) / "images" / "train"
    for img_path in img_dir_path.glob("*.jpg"):

        if random.random() < split_ratio and files_moved < limit:
            files_moved += 1
            new_img_dir_path = img_dir_path.parent / "val"
            new_img_dir_path.mkdir(exist_ok=True)
            img_path.rename(new_img_dir_path / img_path.name)

            label_file_path = (
                img_dir_path.parent.parent
                / "labels"
                / "train"
                / (img_path.stem + ".txt")
            )
            new_label_dir_path = label_file_path.parent.parent / "val"
            new_label_dir_path.mkdir(exist_ok=True)
            label_file_path.rename(new_label_dir_path / label_file_path.name)

            print(f"Moving {img_path} to {new_img_dir_path / img_path.name}")
            print(
                f"Moving {label_file_path} to {new_label_dir_path / label_file_path.name}"
            )

        else:
            files_not_moved += 1

    print(f"Files moved: {files_moved}")
    print(f"Files not moved: {files_not_moved}")
    print(
        f"Validation split ratio: {files_moved / (files_moved + files_not_moved):.2f}"
    )


if __name__ == "__main__":
    split_dataset(
        base_dir="/home/bachelor/ml-carbucks/data/final_carbucks/<dsadasdasdasdasda>",
        split_ratio=0.2,
    )
