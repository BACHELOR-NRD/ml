import os
from PIL import Image, ExifTags

LOG_FILE = "ignore_image_issues.txt"


def find_image_issues(img_dir):
    corrupted = []
    exif_issues = []
    empty_images = []
    mode_issues = []
    aspect_issues = []

    for root, _, files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, f)
                try:
                    # Corruption check
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        # Check EXIF orientation
                        try:
                            exif = img.getexif()
                            orientation_tag = next(
                                (
                                    k
                                    for k, v in ExifTags.TAGS.items()
                                    if v == "Orientation"
                                ),
                                None,
                            )
                            if orientation_tag and exif.get(orientation_tag, 1) != 1:
                                exif_issues.append(path)
                        except Exception:
                            exif_issues.append(path)

                        # Check zero-size
                        if img.width == 0 or img.height == 0:
                            empty_images.append(path)

                        # Check mode (ensure RGB)
                        if img.mode != "RGB":
                            mode_issues.append(path)

                        # Check extreme aspect ratios
                        aspect_ratio = img.width / max(1, img.height)
                        if (
                            aspect_ratio < 0.1 or aspect_ratio > 10
                        ):  # arbitrary extremes
                            aspect_issues.append(path)

                except Exception:
                    print(f"Corrupted: {path}")
                    corrupted.append(path)

    # Save results
    with open(LOG_FILE, "w") as log:
        if corrupted:
            log.write("Corrupted images:\n")
            for path in corrupted:
                log.write(path + "\n")
        if exif_issues:
            log.write("\nImages with non-standard EXIF orientation:\n")
            for path in exif_issues:
                log.write(path + "\n")
        if empty_images:
            log.write("\nImages with zero width/height:\n")
            for path in empty_images:
                log.write(path + "\n")
        if mode_issues:
            log.write("\nImages with non-RGB mode:\n")
            for path in mode_issues:
                log.write(path + "\n")
        if aspect_issues:
            log.write("\nImages with extreme aspect ratios:\n")
            for path in aspect_issues:
                log.write(path + "\n")

    print(f"Found {len(corrupted)} corrupted images.")
    print(f"Found {len(exif_issues)} images with EXIF issues.")
    print(f"Found {len(empty_images)} zero-size images.")
    print(f"Found {len(mode_issues)} images with non-RGB mode.")
    print(f"Found {len(aspect_issues)} images with extreme aspect ratios.")
    print(f"Details saved to {LOG_FILE}")


if __name__ == "__main__":
    image_directory = (
        "/home/bachelor/ml-carbucks/data/carbucks/images/train"  # Change this
    )
    find_image_issues(image_directory)
