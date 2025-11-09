import os
from pathlib import Path
from PIL import Image, ImageOps

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


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
            logger.warning(f"⚠️ Failed to process {path}: {e}")

    logger.info(f"✅ Fixed orientation in {fixed_count} image(s) under '{dir_path}'")


def fix_corrupt_images(
    img_dir: str,
    log_file_corrupt: str = "corrupted_images.log",
    log_file_fixed: str = "fixed_images.log",
    jpeg_quality: int = 95,
    valid_extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> None:
    corrupted_images = []
    fixed_images = []

    for root, _, files in os.walk(img_dir):
        for f in files:
            if not f.lower().endswith(valid_extensions):
                continue

            path = os.path.join(root, f)
            try:
                with Image.open(path) as img:
                    # Verify header
                    img.verify()

                # Reopen to fully load for EXIF fix and resave
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                    img.save(path, "JPEG", quality=jpeg_quality)
                    fixed_images.append(path)

            except Exception as e:
                logger.warning(f"[CORRUPT] {path} | {e}")
                corrupted_images.append(path)

    # Write logs
    if corrupted_images:
        with open(log_file_corrupt, "w") as log:
            for path in corrupted_images:
                log.write(path + "\n")
        logger.info(f"Corrupted images logged to {log_file_corrupt}")

    if fixed_images:
        with open(log_file_fixed, "w") as log:
            for path in fixed_images:
                log.write(path + "\n")
        logger.info(f"Fixed images logged to {log_file_fixed}")

    logger.info(f"Total images processed: {len(fixed_images) + len(corrupted_images)}")
    logger.info(
        f"Corrupted: {len(corrupted_images)} | Fixed/Verified: {len(fixed_images)}"
    )
