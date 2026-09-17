from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments_sci4"
RUNS = ROOT / "runs" / "detect"
DATASET = ROOT / "dataset"

CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

EXISTING_RUNS = {
    "YOLOv5s": RUNS / "compare_yolov5s_80ep",
    "YOLOv8s": RUNS / "compare_yolov8s_80ep",
    "YOLOv8s-BiFPN": RUNS / "compare_yolov8s_bifpn_80ep",
    "YOLOv8s-BiFPN-TSO": RUNS / "compare_yolov8s_bifpn_opt_150ep",
    "YOLOv8m": RUNS / "visdrone_verify_80ep2",
    "YOLOv8m-BiFPN": RUNS / "visdrone_yolov8m_bifpn_80ep",
    "YOLOv8m-BiFPN-TSO": RUNS / "final_yolov8m_bifpn_opt_150ep",
    "YOLOv8l": RUNS / "compare_yolov8l_80ep",
    "YOLOv8l-BiFPN": RUNS / "compare_yolov8l_bifpn_80ep",
    "YOLOv8l-BiFPN-TSO": RUNS / "compare_yolov8l_bifpn_opt_150ep",
    "YOLOv10s": RUNS / "compare_yolov10s_80ep",
}

REVIEW_TO_EXPERIMENTS = [
    {
        "review_point": "BiFPN and TSO are only verified as a combined package.",
        "experiment": "Run a factorial ablation: baseline, BiFPN only, each TSO sub-strategy only, TSO combinations, and BiFPN+TSO.",
        "evidence_file": "tso_ablation_plan.csv",
        "status": "needs_training",
    },
    {
        "review_point": "Lack of independent contribution and interaction analysis for the four TSO modules.",
        "experiment": "Use one-factor and cumulative TSO ablations: SGD, warmup, close_mosaic, mixup. Report deltas versus YOLOv8m.",
        "evidence_file": "tso_ablation_plan.csv",
        "status": "needs_training",
    },
    {
        "review_point": "Comparisons with Faster R-CNN, YOLOv5/8/10, and recent small-object methods are insufficient.",
        "experiment": "Reuse completed YOLOv5s, YOLOv8s/m/l, YOLOv10s, and BiFPN scale generalization runs.",
        "evidence_file": "existing_results_summary.csv",
        "status": "completed_from_existing_logs",
    },
    {
        "review_point": "Dataset scenario, small-object ratio, occlusion, and truncation are not quantified.",
        "experiment": "Compute VisDrone object size, class, occlusion, truncation, and per-image difficulty statistics.",
        "evidence_file": "dataset_stats_summary.csv",
        "status": "completed_by_this_script",
    },
    {
        "review_point": "Night/extreme occlusion and failure cases are not analyzed.",
        "experiment": "Create validation subsets for tiny-object-heavy, occluded, truncated, and crowded images; validate key models on these subsets.",
        "evidence_file": "val_subsets/*.txt",
        "status": "subset_lists_ready; validation_needs_gpu_or_time",
    },
    {
        "review_point": "BiFPN replacement count and design rationale lack support.",
        "experiment": "Compare YOLOv8s/m/l baseline, BiFPN only, and BiFPN+TSO to show scale-dependent effect.",
        "evidence_file": "bifpn_generalization_summary.csv",
        "status": "completed_from_existing_logs",
    },
]


def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
        return rows


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summarize_run(name: str, path: Path) -> dict[str, object] | None:
    csv_path = path / "results.csv"
    if not csv_path.exists():
        return None

    rows = read_results_csv(csv_path)
    if not rows:
        return None

    best50 = max(rows, key=lambda r: f(r, "metrics/mAP50(B)"))
    best95 = max(rows, key=lambda r: f(r, "metrics/mAP50-95(B)"))
    last = rows[-1]

    return {
        "model": name,
        "run_dir": str(path),
        "best_mAP50_epoch": int(float(best50["epoch"])),
        "precision": f(best50, "metrics/precision(B)"),
        "recall": f(best50, "metrics/recall(B)"),
        "mAP50": f(best50, "metrics/mAP50(B)"),
        "mAP50-95": f(best50, "metrics/mAP50-95(B)"),
        "best_mAP50-95_epoch": int(float(best95["epoch"])),
        "precision_at_best95": f(best95, "metrics/precision(B)"),
        "recall_at_best95": f(best95, "metrics/recall(B)"),
        "mAP50_at_best95": f(best95, "metrics/mAP50(B)"),
        "best_mAP50-95": f(best95, "metrics/mAP50-95(B)"),
        "last_epoch": int(float(last["epoch"])),
        "weights_best_exists": (path / "weights" / "best.pt").exists(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_image_size(image_path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None


def parse_visdrone_annotations(split: str) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    ann_dir = DATASET / f"VisDrone2019-DET-{split}" / "annotations"
    img_dir = DATASET / f"VisDrone2019-DET-{split}" / "images"
    objects: list[dict[str, object]] = []
    image_stats: dict[str, dict[str, object]] = {}

    for ann_path in sorted(ann_dir.glob("*.txt")):
        image_name = ann_path.with_suffix(".jpg").name
        image_path = img_dir / image_name
        size = read_image_size(image_path)
        if size is None:
            continue
        img_w, img_h = size
        stat = {
            "image": image_name,
            "objects": 0,
            "tiny_objects": 0,
            "small_objects": 0,
            "occluded_objects": 0,
            "severe_occluded_objects": 0,
            "truncated_objects": 0,
            "crowd_score": 0,
        }

        for raw in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 8:
                continue
            left, top, width, height, score, cls, truncation, occlusion = map(float, parts[:8])
            if int(score) == 0 or int(cls) < 1 or int(cls) > len(CLASSES):
                continue
            area = width * height
            rel_area = area / max(1.0, img_w * img_h)
            sqrt_area = math.sqrt(max(0.0, area))
            if rel_area < 0.001:
                size_bin = "tiny_rel_lt_0.1pct"
            elif rel_area < 0.01:
                size_bin = "small_rel_0.1_1pct"
            elif rel_area < 0.05:
                size_bin = "medium_rel_1_5pct"
            else:
                size_bin = "large_rel_ge_5pct"

            obj = {
                "split": split,
                "image": image_name,
                "class_id": int(cls) - 1,
                "class_name": CLASSES[int(cls) - 1],
                "width_px": width,
                "height_px": height,
                "area_px": area,
                "sqrt_area_px": sqrt_area,
                "relative_area": rel_area,
                "size_bin": size_bin,
                "truncation": int(truncation),
                "occlusion": int(occlusion),
            }
            objects.append(obj)

            stat["objects"] += 1
            stat["crowd_score"] += 1
            if rel_area < 0.001:
                stat["tiny_objects"] += 1
            if rel_area < 0.01:
                stat["small_objects"] += 1
            if int(occlusion) > 0:
                stat["occluded_objects"] += 1
            if int(occlusion) >= 2:
                stat["severe_occluded_objects"] += 1
            if int(truncation) > 0:
                stat["truncated_objects"] += 1

        image_stats[image_name] = stat

    return objects, image_stats


def summarize_dataset() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    all_objects: list[dict[str, object]] = []
    all_images: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []

    for split in ["train", "val"]:
        objects, images = parse_visdrone_annotations(split)
        all_objects.extend(objects)
        all_images.update({f"{split}/{k}": v | {"split": split} for k, v in images.items()})
        class_counter = Counter(o["class_name"] for o in objects)
        size_counter = Counter(o["size_bin"] for o in objects)
        occlusion_counter = Counter(o["occlusion"] for o in objects)
        trunc_counter = Counter(o["truncation"] for o in objects)
        rel_areas = [float(o["relative_area"]) for o in objects]
        sqrt_areas = [float(o["sqrt_area_px"]) for o in objects]

        total = len(objects)
        summary_rows.append(
            {
                "split": split,
                "images": len(images),
                "objects": total,
                "objects_per_image": round(total / max(1, len(images)), 4),
                "tiny_rel_lt_0.1pct": size_counter["tiny_rel_lt_0.1pct"],
                "tiny_ratio": round(size_counter["tiny_rel_lt_0.1pct"] / max(1, total), 6),
                "small_rel_lt_1pct": size_counter["tiny_rel_lt_0.1pct"] + size_counter["small_rel_0.1_1pct"],
                "small_ratio": round(
                    (size_counter["tiny_rel_lt_0.1pct"] + size_counter["small_rel_0.1_1pct"]) / max(1, total), 6
                ),
                "occluded_objects": sum(v for k, v in occlusion_counter.items() if int(k) > 0),
                "occluded_ratio": round(sum(v for k, v in occlusion_counter.items() if int(k) > 0) / max(1, total), 6),
                "severe_occluded_objects": sum(v for k, v in occlusion_counter.items() if int(k) >= 2),
                "truncated_objects": sum(v for k, v in trunc_counter.items() if int(k) > 0),
                "truncated_ratio": round(sum(v for k, v in trunc_counter.items() if int(k) > 0) / max(1, total), 6),
                "median_relative_area": round(statistics.median(rel_areas), 8) if rel_areas else 0,
                "median_sqrt_area_px": round(statistics.median(sqrt_areas), 4) if sqrt_areas else 0,
                "top_class": class_counter.most_common(1)[0][0] if class_counter else "",
            }
        )

    return summary_rows, all_images


def make_val_subsets(image_stats: dict[str, dict[str, object]]) -> None:
    subset_dir = OUT / "val_subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    val_images = [v for k, v in image_stats.items() if str(k).startswith("val/")]

    def rank(metric: str, min_objects: int = 1) -> list[dict[str, object]]:
        return sorted(
            [x for x in val_images if int(x["objects"]) >= min_objects],
            key=lambda x: (int(x[metric]), int(x["objects"])),
            reverse=True,
        )

    subsets = {
        "tiny_heavy_top100.txt": rank("tiny_objects")[:100],
        "small_heavy_top100.txt": rank("small_objects")[:100],
        "occlusion_top100.txt": rank("occluded_objects")[:100],
        "severe_occlusion_top100.txt": rank("severe_occluded_objects")[:100],
        "truncation_top100.txt": rank("truncated_objects")[:100],
        "crowded_top100.txt": rank("crowd_score")[:100],
    }

    img_root = DATASET / "VisDrone2019-DET-val" / "images"
    for filename, rows in subsets.items():
        lines = [str((img_root / str(row["image"])).resolve()) for row in rows]
        (subset_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    subset_summary = [
        {
            "subset": name,
            "images": len(rows),
            "objects": sum(int(r["objects"]) for r in rows),
            "tiny_objects": sum(int(r["tiny_objects"]) for r in rows),
            "small_objects": sum(int(r["small_objects"]) for r in rows),
            "occluded_objects": sum(int(r["occluded_objects"]) for r in rows),
            "severe_occluded_objects": sum(int(r["severe_occluded_objects"]) for r in rows),
            "truncated_objects": sum(int(r["truncated_objects"]) for r in rows),
        }
        for name, rows in subsets.items()
    ]
    write_csv(OUT / "val_subsets_summary.csv", subset_summary)


def make_tso_plan() -> None:
    base = {
        "epochs": 80,
        "imgsz": 1024,
        "batch": 8,
        "model_yaml": "src/yolov8m.yaml_or_pretrained_yolov8m.pt",
        "pretrained": "src/yolov8m.pt",
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "warmup_epochs": 0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "mosaic": 1.0,
        "cos_lr": False,
        "bifpn": False,
    }
    experiments = [
        ("A0_yolov8m_baseline_recheck", {}),
        ("A1_sgd_only", {"optimizer": "SGD", "lr0": 0.005, "cos_lr": True}),
        ("A2_warmup_only", {"warmup_epochs": 5}),
        ("A3_close_mosaic_only", {"close_mosaic": 15}),
        ("A4_mixup_only", {"mixup": 0.1}),
        ("A5_sgd_warmup", {"optimizer": "SGD", "lr0": 0.005, "cos_lr": True, "warmup_epochs": 5}),
        ("A6_sgd_warmup_close_mosaic", {"optimizer": "SGD", "lr0": 0.005, "cos_lr": True, "warmup_epochs": 5, "close_mosaic": 15}),
        ("A7_tso_all_no_bifpn", {"optimizer": "SGD", "lr0": 0.005, "cos_lr": True, "warmup_epochs": 5, "close_mosaic": 15, "mixup": 0.1}),
        ("B0_bifpn_only", {"bifpn": True, "model_yaml": "src/yolov8m-bifpn.yaml"}),
        ("B1_bifpn_sgd", {"bifpn": True, "model_yaml": "src/yolov8m-bifpn.yaml", "optimizer": "SGD", "lr0": 0.005, "cos_lr": True}),
        ("B2_bifpn_warmup", {"bifpn": True, "model_yaml": "src/yolov8m-bifpn.yaml", "warmup_epochs": 5}),
        ("B3_bifpn_close_mosaic", {"bifpn": True, "model_yaml": "src/yolov8m-bifpn.yaml", "close_mosaic": 15}),
        ("B4_bifpn_mixup", {"bifpn": True, "model_yaml": "src/yolov8m-bifpn.yaml", "mixup": 0.1}),
        ("B5_bifpn_tso_all_150ep", {"bifpn": True, "model_yaml": "src/yolov8m-bifpn.yaml", "epochs": 150, "optimizer": "SGD", "lr0": 0.005, "cos_lr": True, "warmup_epochs": 5, "close_mosaic": 15, "mixup": 0.1}),
    ]
    rows = []
    for name, override in experiments:
        row = {"experiment": name} | base | override
        row["output_name"] = f"sci4_{name}"
        rows.append(row)
    write_csv(OUT / "tso_ablation_plan.csv", rows)
    (OUT / "tso_ablation_plan.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def make_bifpn_generalization(rows: list[dict[str, object]]) -> None:
    by_name = {str(r["model"]): r for r in rows}
    out_rows = []
    for scale in ["s", "m", "l"]:
        base = by_name.get(f"YOLOv8{scale}")
        bifpn = by_name.get(f"YOLOv8{scale}-BiFPN")
        opt = by_name.get(f"YOLOv8{scale}-BiFPN-TSO")
        if not base:
            continue
        for candidate, tag in [(bifpn, "BiFPN"), (opt, "BiFPN-TSO")]:
            if not candidate:
                continue
            out_rows.append(
                {
                    "scale": scale,
                    "baseline": base["model"],
                    "variant": candidate["model"],
                    "variant_type": tag,
                    "delta_precision": round(float(candidate["precision"]) - float(base["precision"]), 6),
                    "delta_recall": round(float(candidate["recall"]) - float(base["recall"]), 6),
                    "delta_mAP50": round(float(candidate["mAP50"]) - float(base["mAP50"]), 6),
                    "delta_mAP50-95": round(float(candidate["mAP50-95"]) - float(base["mAP50-95"]), 6),
                    "relative_mAP50_gain_pct": round((float(candidate["mAP50"]) - float(base["mAP50"])) / float(base["mAP50"]) * 100, 4),
                }
            )
    write_csv(OUT / "bifpn_generalization_summary.csv", out_rows)


def write_review_plan() -> None:
    write_csv(OUT / "review_to_experiment_map.csv", REVIEW_TO_EXPERIMENTS)
    md = [
        "# SCI4 revision experiment checklist",
        "",
        "This file maps the rejection comments to concrete experiments and generated evidence files.",
        "",
    ]
    for idx, row in enumerate(REVIEW_TO_EXPERIMENTS, 1):
        md.extend(
            [
                f"## {idx}. {row['review_point']}",
                f"- Experiment: {row['experiment']}",
                f"- Evidence: `{row['evidence_file']}`",
                f"- Status: `{row['status']}`",
                "",
            ]
        )
    (OUT / "review_to_experiment_map.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    result_rows = [r for name, path in EXISTING_RUNS.items() if (r := summarize_run(name, path))]
    write_csv(OUT / "existing_results_summary.csv", result_rows)
    make_bifpn_generalization(result_rows)

    dataset_rows, image_stats = summarize_dataset()
    write_csv(OUT / "dataset_stats_summary.csv", dataset_rows)
    write_csv(OUT / "val_image_difficulty.csv", list(image_stats.values()))
    make_val_subsets(image_stats)

    make_tso_plan()
    write_review_plan()

    print(f"Wrote SCI4 experiment package to: {OUT}")
    print("Key outputs:")
    for rel in [
        "review_to_experiment_map.md",
        "existing_results_summary.csv",
        "bifpn_generalization_summary.csv",
        "dataset_stats_summary.csv",
        "val_image_difficulty.csv",
        "val_subsets_summary.csv",
        "tso_ablation_plan.csv",
    ]:
        print(f"  - {OUT / rel}")


if __name__ == "__main__":
    main()
