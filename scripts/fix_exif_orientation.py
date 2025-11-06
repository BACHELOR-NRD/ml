import argparse
from pathlib import Path
from PIL import Image, ImageOps


def fix_exif_orientation_in_dir(dir_path: str | Path, recursive: bool = True) -> None:
    """
    Fix EXIF orientation for all images in a directory by rewriting them
    with pixel data correctly rotated (removes EXIF orientation flag).

    Args:
        dir_path (str | Path): Path to the directory containing images.
        recursive (bool): Whether to process subdirectories recursively.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    dir_path = Path(dir_path)
    files = dir_path.rglob("*") if recursive else dir_path.glob("*")

    fixed_count = 0
    for path in files:
        if path.suffix.lower() not in img_exts:
            continue

        try:
            with Image.open(path) as img:
                # Get EXIF orientation flag if present
                exif = img.getexif()
                orientation = exif.get(274, 1)  # 274 is the EXIF tag for orientation

                if orientation != 1:
                    img_corrected = ImageOps.exif_transpose(img)
                    img_corrected.save(path, quality=95)
                    fixed_count += 1
        except Exception as e:
            print(f"⚠️ Failed to process {path}: {e}")

    print(f"✅ Fixed orientation in {fixed_count} image(s) under '{dir_path}'")


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

    fix_exif_orientation_in_dir(args.dir_path, args.recursive)
