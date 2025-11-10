#!/usr/bin/env python3
"""Generate descriptive statistics for COCO-style annotation files.

The script scans a directory of annotation JSON files, computes several
dataset-level and category-level metrics (counts, percentages, bbox stats),
and stores everything in a CSV that lives next to this script by default.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute statistics for COCO annotation files and export CSV."
    )
    default_annotations = Path(__file__).parent / "data" / "backup_carbucks" / "annotations"
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=default_annotations,
        help=f"Directory containing *.json annotation files (default: {default_annotations})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "coco_stats.csv",
        help="CSV output path (default: coco_stats.csv next to this script).",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(__file__).parent / "coco_stats.html",
        help="Optional HTML output path for viewing in a browser (default: coco_stats.html).",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(__file__).parent / "coco_stats.md",
        help="Optional Markdown output path (default: coco_stats.md).",
    )
    parser.add_argument(
        "--category-summary-output",
        type=Path,
        default=Path(__file__).parent / "coco_category_summary.txt",
        help="Optional text file with aggregated category counts and percentages.",
    )
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


@dataclass
class DatasetComputation:
    summary_row: Dict[str, float | int | str]
    category_rows: List[Dict[str, float | int | str]]
    annotations_per_image: List[int]
    bbox_areas: List[float]
    bbox_area_pcts: List[float]
    bbox_widths: List[float]
    bbox_heights: List[float]
    total_image_area: float
    category_counts_by_name: Dict[str, int]


def compute_dataset_stats(path: Path) -> DatasetComputation:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    category_lookup = {cat["id"]: cat.get("name", str(cat["id"])) for cat in categories}

    images_by_id: Dict[int, dict] = {img["id"]: img for img in images}
    ann_per_image: Dict[int, List[dict]] = defaultdict(list)
    category_counts: Counter[int] = Counter()

    bbox_areas: List[float] = []
    bbox_area_percentages: List[float] = []
    bbox_widths: List[float] = []
    bbox_heights: List[float] = []

    for ann in annotations:
        image_id = ann.get("image_id")
        ann_per_image[image_id].append(ann)
        category_counts[ann.get("category_id")] += 1

        bbox = ann.get("bbox") or [0, 0, 0, 0]
        width = float(bbox[2]) if len(bbox) >= 3 else 0.0
        height = float(bbox[3]) if len(bbox) >= 4 else 0.0
        area = max(width, 0.0) * max(height, 0.0)
        bbox_widths.append(width)
        bbox_heights.append(height)
        bbox_areas.append(area)

        image = images_by_id.get(image_id, {})
        img_width = image.get("width", 0) or 0
        img_height = image.get("height", 0) or 0
        img_area = img_width * img_height
        bbox_area_percentages.append(safe_div(area, img_area) * 100 if img_area else 0.0)

    num_images = len(images)
    num_annotations = len(annotations)
    num_categories = len(categories)
    images_with_annotations = len(ann_per_image)
    images_without_annotations = max(num_images - images_with_annotations, 0)

    annotations_per_image = [
        len(ann_per_image.get(image_id, [])) for image_id in images_by_id.keys()
    ]
    annotated_counts = [count for count in annotations_per_image if count > 0]

    category_usage_pct = safe_div(
        sum(1 for count in category_counts.values() if count > 0),
        num_categories,
    ) * 100 if num_categories else 0.0

    if category_counts:
        dominant_category_id, dominant_count = category_counts.most_common(1)[0]
        dominant_category_name = category_lookup.get(dominant_category_id, str(dominant_category_id))
    else:
        dominant_category_name = ""
        dominant_count = 0

    total_image_area = sum(
        (img.get("width", 0) or 0) * (img.get("height", 0) or 0) for img in images
    )
    bbox_area_sum = sum(bbox_areas)

    summary_row = {
        "dataset": path.name,
        "row_type": "summary",
        "category": "",
        "total_images": num_images,
        "total_image_area_px": total_image_area,
        "images_with_annotations": images_with_annotations,
        "images_without_annotations": images_without_annotations,
        "pct_images_with_annotations": safe_div(images_with_annotations, num_images) * 100 if num_images else 0.0,
        "pct_images_without_annotations": safe_div(images_without_annotations, num_images) * 100 if num_images else 0.0,
        "total_annotations": num_annotations,
        "avg_ann_per_image": safe_div(num_annotations, num_images),
        "median_ann_per_image": median(annotations_per_image),
        "avg_ann_per_annotated_image": safe_div(sum(annotated_counts), len(annotated_counts)),
        "num_categories_defined": num_categories,
        "category_usage_pct": category_usage_pct,
        "dominant_category": dominant_category_name,
        "dominant_category_pct": safe_div(dominant_count, num_annotations) * 100 if num_annotations else 0.0,
        "total_bbox_area": bbox_area_sum,
        "avg_bbox_area": safe_div(bbox_area_sum, len(bbox_areas)),
        "median_bbox_area": median(bbox_areas),
        "avg_bbox_width": safe_div(sum(bbox_widths), len(bbox_widths)),
        "avg_bbox_height": safe_div(sum(bbox_heights), len(bbox_heights)),
        "avg_bbox_area_pct_of_image": safe_div(sum(bbox_area_percentages), len(bbox_area_percentages)),
        "median_bbox_area_pct_of_image": median(bbox_area_percentages),
        "total_bbox_area_pct_of_all_images": safe_div(bbox_area_sum, total_image_area) * 100 if total_image_area else 0.0,
        "category_annotation_pct": "",
    }

    category_rows: List[Dict[str, float | int | str]] = []
    category_counts_by_name: Dict[str, int] = defaultdict(int)
    for category_id, defined_category in sorted(category_lookup.items(), key=lambda item: item[1]):
        count = category_counts.get(category_id, 0)
        category_counts_by_name[defined_category] += count
        category_rows.append(
            {
                "dataset": path.name,
                "row_type": "category_breakdown",
                "category": defined_category,
                "total_images": num_images,
                "total_image_area_px": total_image_area,
                "images_with_annotations": images_with_annotations,
                "images_without_annotations": images_without_annotations,
                "pct_images_with_annotations": safe_div(images_with_annotations, num_images) * 100 if num_images else 0.0,
                "pct_images_without_annotations": safe_div(images_without_annotations, num_images) * 100 if num_images else 0.0,
                "total_annotations": count,
                "avg_ann_per_image": safe_div(count, num_images),
                "median_ann_per_image": "",
                "avg_ann_per_annotated_image": "",
                "num_categories_defined": num_categories,
                "category_usage_pct": "",
                "dominant_category": "",
                "dominant_category_pct": "",
                "total_bbox_area": "",
                "avg_bbox_area": "",
                "median_bbox_area": "",
                "avg_bbox_width": "",
                "avg_bbox_height": "",
                "avg_bbox_area_pct_of_image": "",
                "median_bbox_area_pct_of_image": "",
                "total_bbox_area_pct_of_all_images": "",
                "category_annotation_pct": safe_div(count, num_annotations) * 100 if num_annotations else 0.0,
            }
        )

    return DatasetComputation(
        summary_row=summary_row,
        category_rows=category_rows,
        annotations_per_image=annotations_per_image,
        bbox_areas=bbox_areas,
        bbox_area_pcts=bbox_area_percentages,
        bbox_widths=bbox_widths,
        bbox_heights=bbox_heights,
        total_image_area=total_image_area,
        category_counts_by_name=dict(category_counts_by_name),
    )


def aggregate_overall(rows: List[DatasetComputation]) -> Dict[str, float | int | str]:
    total_images = sum(row.summary_row["total_images"] for row in rows)
    images_with_annotations = sum(row.summary_row["images_with_annotations"] for row in rows)
    images_without_annotations = sum(row.summary_row["images_without_annotations"] for row in rows)
    total_annotations = sum(row.summary_row["total_annotations"] for row in rows)
    total_bbox_area = sum(row.summary_row["total_bbox_area"] for row in rows)
    total_image_area = sum(row.total_image_area for row in rows)

    annotations_per_image = [count for row in rows for count in row.annotations_per_image]
    annotated_counts = [count for count in annotations_per_image if count > 0]
    bbox_areas = [value for row in rows for value in row.bbox_areas]
    bbox_area_pcts = [value for row in rows for value in row.bbox_area_pcts]
    bbox_widths = [value for row in rows for value in row.bbox_widths]
    bbox_heights = [value for row in rows for value in row.bbox_heights]

    return {
        "dataset": "ALL",
        "row_type": "summary",
        "category": "",
        "total_images": total_images,
        "total_image_area_px": total_image_area,
        "images_with_annotations": images_with_annotations,
        "images_without_annotations": images_without_annotations,
        "pct_images_with_annotations": safe_div(images_with_annotations, total_images) * 100 if total_images else 0.0,
        "pct_images_without_annotations": safe_div(images_without_annotations, total_images) * 100 if total_images else 0.0,
        "total_annotations": total_annotations,
        "avg_ann_per_image": safe_div(total_annotations, total_images),
        "median_ann_per_image": median(annotations_per_image),
        "avg_ann_per_annotated_image": safe_div(sum(annotated_counts), len(annotated_counts)),
        "num_categories_defined": "",
        "category_usage_pct": "",
        "dominant_category": "",
        "dominant_category_pct": "",
        "total_bbox_area": total_bbox_area,
        "avg_bbox_area": safe_div(sum(bbox_areas), len(bbox_areas)),
        "median_bbox_area": median(bbox_areas),
        "avg_bbox_width": safe_div(sum(bbox_widths), len(bbox_widths)),
        "avg_bbox_height": safe_div(sum(bbox_heights), len(bbox_heights)),
        "avg_bbox_area_pct_of_image": safe_div(sum(bbox_area_pcts), len(bbox_area_pcts)),
        "median_bbox_area_pct_of_image": median(bbox_area_pcts),
        "total_bbox_area_pct_of_all_images": safe_div(total_bbox_area, total_image_area) * 100 if total_image_area else 0.0,
        "category_annotation_pct": "",
    }


def aggregate_category_counts(rows: List[DatasetComputation]) -> Counter[str]:
    total = Counter()
    for row in rows:
        total.update(row.category_counts_by_name)
    return total


def write_html_table(
    rows: List[Dict[str, float | int | str]],
    fieldnames: Sequence[str],
    output_path: Path,
) -> None:
    """Render the stats rows into a simple HTML table for quick inspection."""
    styles = """
    body { font-family: Arial, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 0.4rem; text-align: right; }
    th { background: #f3f3f3; position: sticky; top: 0; }
    td:first-child, th:first-child { text-align: left; }
    tr.summary { background: #fdf7f2; }
    tr.category_breakdown { background: #f9fbff; }
    caption { text-align: left; font-weight: bold; margin-bottom: 0.5rem; }
    """
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<title>COCO Stats</title>",
        f"<style>{styles}</style>",
        "</head>",
        "<body>",
        "<h1>COCO Annotation Statistics</h1>",
        "<table>",
        "<caption>All datasets and per-category breakdown</caption>",
        "<thead>",
        "<tr>",
    ]

    for header in fieldnames:
        lines.append(f"<th>{html.escape(header)}</th>")
    lines.extend(["</tr>", "</thead>", "<tbody>"])

    for row in rows:
        row_type = row.get("row_type", "")
        css_class = f" class='{row_type}'" if row_type else ""
        lines.append(f"<tr{css_class}>")
        for field in fieldnames:
            value = row.get(field, "")
            display = "" if value is None else f"{value}"
            lines.append(f"<td>{html.escape(display)}</td>")
        lines.append("</tr>")

    lines.extend(["</tbody>", "</table>", "</body>", "</html>"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_category_summary(
    category_counts: Counter[str],
    total_annotations: int,
    output_path: Path,
) -> None:
    lines = [
        "COCO Annotation Category Summary",
        "=" * 36,
        f"Total annotations: {total_annotations}",
        "",
        "Breakdown by category:",
    ]

    for name, count in category_counts.most_common():
        pct = safe_div(count, total_annotations) * 100 if total_annotations else 0.0
        lines.append(f"- {name}: {count} annotations ({pct:.2f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_table(
    rows: List[Dict[str, float | int | str]],
    fieldnames: Sequence[str],
    output_path: Path,
) -> None:
    """Write a Markdown table so the stats are easy to preview anywhere."""
    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    table_lines = [header, separator]

    for row in rows:
        values = []
        for field in fieldnames:
            value = row.get(field, "")
            if value is None:
                value = ""
            values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")

    content = "# COCO Annotation Statistics\n\n" + "\n".join(table_lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotations_dir: Path = args.annotations_dir
    output_path: Path = args.output
    html_output_path: Path | None = args.html_output
    markdown_output_path: Path | None = args.markdown_output
    category_summary_path: Path | None = args.category_summary_output

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    annotation_files = sorted(annotations_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotation files found in {annotations_dir}")

    computations: List[DatasetComputation] = []
    csv_rows: List[Dict[str, float | int | str]] = []

    for annotation_file in annotation_files:
        stats = compute_dataset_stats(annotation_file)
        computations.append(stats)
        csv_rows.append(stats.summary_row)
        csv_rows.extend(stats.category_rows)

    overall_row = aggregate_overall(computations)
    csv_rows.append(overall_row)
    category_counts = aggregate_category_counts(computations)

    fieldnames = [
        "dataset",
        "row_type",
        "category",
        "total_images",
        "total_image_area_px",
        "images_with_annotations",
        "images_without_annotations",
        "pct_images_with_annotations",
        "pct_images_without_annotations",
        "total_annotations",
        "avg_ann_per_image",
        "median_ann_per_image",
        "avg_ann_per_annotated_image",
        "num_categories_defined",
        "category_usage_pct",
        "dominant_category",
        "dominant_category_pct",
        "total_bbox_area",
        "avg_bbox_area",
        "median_bbox_area",
        "avg_bbox_width",
        "avg_bbox_height",
        "avg_bbox_area_pct_of_image",
        "median_bbox_area_pct_of_image",
        "total_bbox_area_pct_of_all_images",
        "category_annotation_pct",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Wrote stats for {len(annotation_files)} files to {output_path}")

    if html_output_path:
        write_html_table(csv_rows, fieldnames, html_output_path)
        print(f"Wrote browser-friendly table to {html_output_path}")

    if markdown_output_path:
        write_markdown_table(csv_rows, fieldnames, markdown_output_path)
        print(f"Wrote Markdown table to {markdown_output_path}")

    if category_summary_path:
        total_annotations = int(overall_row["total_annotations"])
        write_category_summary(category_counts, total_annotations, category_summary_path)
        print(f"Wrote category summary to {category_summary_path}")


if __name__ == "__main__":
    main()
