"""Convert local ExDark bbGt annotations using the official experimental splits.

Run from the repository root: python tools/prepare_exdark.py
Requires Pillow. Originals are preserved; the destination must not already exist.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup",
    "Dog", "Motorbike", "People", "Table",
)
SPLITS = {1: "train", 2: "val", 3: "test"}


def index_files(root, suffixes):
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = str(path.relative_to(root)).lower()
        if key in files:
            raise ValueError(f"Duplicate filename ignoring case: {path}")
        files[key] = path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "ExDark")
    parser.add_argument("--annotations", type=Path, default=ROOT / "ExDark_Annno")
    parser.add_argument("--split-file", type=Path, default=ROOT / "data/exdark_imageclasslist.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/exdark")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Destination already exists: {output}; use a new --output directory")

    images = index_files(args.images, {".jpg", ".jpeg", ".png", ".bmp"})
    annotations = index_files(args.annotations, {".txt"})
    records, seen_images, seen_annotations, destinations = [], set(), set(), set()
    clipped, orientation_fixed = [], []
    stats = {split: {"images": 0, "boxes": 0, "classes": Counter()} for split in SPLITS.values()}
    # Validate everything before creating the destination.
    for line in args.split_file.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, class_id, _light, _scene, split_id = line.split()
        category = CLASSES[int(class_id) - 1]
        split = SPLITS[int(split_id)]
        key = f"{category}/{name}".lower()
        image, annotation = images[key], annotations[key + ".txt"]
        with Image.open(image) as img:
            img.verify()
        with Image.open(image) as img:
            width, height = img.size
            reset_orientation = img.getexif().get(274, 1) in range(2, 9)
        # Annotations use the raw raster coordinates. Prevent YOLO's EXIF auto-rotation
        # by exporting affected images losslessly as PNG without orientation metadata.
        suffix = ".png" if reset_orientation else image.suffix.lower()
        dest_name = f"{category}_{image.stem}{suffix}"
        dest_key = Path(dest_name).stem
        if key in seen_images or dest_key in destinations:
            raise ValueError(f"Duplicate image or output name: {image}")
        seen_images.add(key)
        seen_annotations.add(key + ".txt")
        destinations.add(dest_key)
        if reset_orientation:
            orientation_fixed.append(str(image.relative_to(args.images)))
        lines = annotation.read_text().splitlines()
        if not lines or lines[0].strip() != "% bbGt version=3":
            raise ValueError(f"Unexpected annotation header: {annotation}")
        labels = []
        for row, text in enumerate(lines[1:], start=2):
            if not text.strip():
                continue
            fields = text.split()
            if len(fields) != 12:
                raise ValueError(f"Invalid bbGt row: {annotation}:{row}")
            label = CLASSES.index(fields[0])
            x, y, w, h = map(float, fields[1:5])
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f"Invalid box: {annotation}:{row}")
            # bbGt stores left, top, width, height in pixels. Clip to image bounds.
            raw = (x, y, x + w, y + h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Box outside image: {annotation}:{row}")
            if raw != (x1, y1, x2, y2):
                clipped.append({"annotation": str(annotation.relative_to(args.annotations)),
                                "row": row, "original_xyxy": raw,
                                "clipped_xyxy": (x1, y1, x2, y2)})
            values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                      (x2 - x1) / width, (y2 - y1) / height)
            labels.append(f"{label} " + " ".join(f"{v:.10f}" for v in values))
            stats[split]["classes"][CLASSES[label]] += 1
        stats[split]["images"] += 1
        stats[split]["boxes"] += len(labels)
        records.append((image, split, dest_name, labels, reset_orientation))
    if seen_images != set(images) or seen_annotations != set(annotations):
        raise ValueError("Official split does not cover every local image/annotation")

    for split in SPLITS.values():
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for image, split, name, labels, reset_orientation in records:
        destination = output / "images" / split / name
        if reset_orientation:
            with Image.open(image) as img:
                img.save(destination, format="PNG", exif=b"")
        else:
            shutil.copy2(image, destination)
        (output / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "".join(line + "\n" for line in labels)
        )
    (output / "data.yaml").write_text(
        f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        + f"nc: {len(CLASSES)}\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )
    report = {
        "split_source": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/imageclasslist.txt",
        "coordinate_conversion": "bbGt xywh -> clipped xyxy -> normalized YOLO center xywh",
        "splits": stats, "clipped_boxes": clipped,
        "orientation_metadata_removed": orientation_fixed,
    }
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"Clipped boxes: {len(clipped)}; dataset: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
