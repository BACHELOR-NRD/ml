import os
import sys
from PIL import Image
import imagehash
import json


def find_duplicate_images(folder):
    hashes = {}
    duplicates = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
            ):
                filepath = os.path.join(root, file)
                try:
                    img = Image.open(filepath)
                    h = imagehash.average_hash(img)  # You can also try phash or dhash
                    img.close()

                    if h in hashes:
                        print(f"[DUPLICATE] {filepath} == {hashes[h]}")
                        duplicates.append((filepath, hashes[h]))
                    else:
                        hashes[h] = filepath

                except Exception as e:
                    print(f"[ERROR] {filepath}: {e}")

    if not duplicates:
        print("✅ No duplicates found.")
    else:
        print(f"🟠 Found {len(duplicates)} duplicate pairs.")

    return duplicates


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 find_duplicate_images.py /path/to/folder")
    else:
        dd = find_duplicate_images(sys.argv[1])
        with open(f"duplicate_images_{os.path.basename(sys.argv[1])}.json", "w") as f:
            json.dump(dd, f, indent=4)
