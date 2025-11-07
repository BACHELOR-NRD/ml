import os
from PIL import Image, ImageOps

# ---------------- CONFIG ---------------- #
IMAGE_DIR = ""  # Change this to your dataset directory
LOG_FILE_CORRUPT = "corrupted_images.log"
LOG_FILE_FIXED = "fixed_images.log"
JPEG_QUALITY = 95
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
# ---------------------------------------- #

corrupted_images = []
fixed_images = []


def process_images(img_dir: str):
    for root, _, files in os.walk(img_dir):
        for f in files:
            if not f.lower().endswith(VALID_EXTENSIONS):
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
                    img.save(path, "JPEG", quality=JPEG_QUALITY)
                    fixed_images.append(path)

            except Exception as e:
                print(f"[CORRUPT] {path} | {e}")
                corrupted_images.append(path)

    # Write logs
    if corrupted_images:
        with open(LOG_FILE_CORRUPT, "w") as log:
            for path in corrupted_images:
                log.write(path + "\n")
        print(f"Corrupted images logged to {LOG_FILE_CORRUPT}")

    if fixed_images:
        with open(LOG_FILE_FIXED, "w") as log:
            for path in fixed_images:
                log.write(path + "\n")
        print(f"Fixed images logged to {LOG_FILE_FIXED}")

    print(f"Total images processed: {len(fixed_images) + len(corrupted_images)}")
    print(f"Corrupted: {len(corrupted_images)} | Fixed/Verified: {len(fixed_images)}")


if __name__ == "__main__":
    process_images(IMAGE_DIR)
