import argparse
from ml_carbucks.utils.data_corrections import fix_exif_orientation_in_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fix EXIF orientation for all images in a directory."
    )
    parser.add_argument(
        "dir_path", type=str, help="Path to the directory containing images."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively.",
    )
    args = parser.parse_args()
    fix_exif_orientation_in_dir(args.dir_path, recursive=args.recursive)
