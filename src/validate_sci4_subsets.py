from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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

MODELS = {
    "YOLOv8m": ROOT / "runs" / "detect" / "visdrone_verify_80ep2" / "weights" / "best.pt",
    "YOLOv8m-BiFPN": ROOT / "runs" / "detect" / "visdrone_yolov8m_bifpn_80ep" / "weights" / "best.pt",
    "YOLOv8m-BiFPN-TSO": ROOT / "runs" / "detect" / "final_yolov8m_bifpn_opt_150ep" / "weights" / "best.pt",
    "YOLOv8l": ROOT / "runs" / "detect" / "compare_yolov8l_80ep" / "weights" / "best.pt",
    "YOLOv10s": ROOT / "runs" / "detect" / "compare_yolov10s_80ep" / "weights" / "best.pt",
}


def set_local_ultralytics_config(root: Path) -> None:
    cfg = root / ".ultralytics"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))


def write_subset_yaml(path: Path, subset_txt: Path, dataset_root: Path) -> Path:
    """Materialize a subset list with paths resolved against ``dataset_root``
    (subset .txt files store paths relative to the VisDrone dataset root) and
    write the matching data yaml."""
    resolved_txt = path / f"{subset_txt.stem}_resolved.txt"
    lines = []
    for line in subset_txt.read_text(encoding="utf-8").split():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if not p.is_absolute():
            p = dataset_root / p
        lines.append(str(p))
    resolved_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_path = path / f"{subset_txt.stem}.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    yaml_path.write_text(
        f"path: {dataset_root}\n"
        f"train: VisDrone2019-DET-train/images\n"
        f"val: {resolved_txt}\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def metric_value(metrics, name: str) -> float:
    box = metrics.box
    return float(getattr(box, name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate models on SCI4 difficulty subsets.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--subsets", type=Path, default=ROOT / "experiments_sci4" / "val_subsets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "val_sci4_subsets")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    args = parser.parse_args()

    root = args.root.resolve()
    set_local_ultralytics_config(root)

    from ultralytics import YOLO

    args.project.mkdir(parents=True, exist_ok=True)
    yaml_dir = root / "experiments_sci4" / "subset_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    subset_files = sorted(args.subsets.glob("*.txt"))
    for subset_txt in subset_files:
        subset_yaml = write_subset_yaml(yaml_dir, subset_txt.resolve(), (root / "dataset").resolve())
        for model_name in args.models:
            weight = MODELS.get(model_name)
            if weight is None or not weight.exists():
                print(f"Skipping missing model: {model_name} -> {weight}")
                continue
            run_name = f"{model_name}_{subset_txt.stem}".replace("/", "_").replace(" ", "_")
            print("=" * 100)
            print(f"Validating {model_name} on {subset_txt.name}")
            print("=" * 100)
            model = YOLO(str(weight))
            metrics = model.val(
                data=str(subset_yaml),
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                project=str(args.project),
                name=run_name,
                exist_ok=True,
                plots=True,
                verbose=True,
            )
            rows.append(
                {
                    "model": model_name,
                    "subset": subset_txt.name,
                    "precision": metric_value(metrics, "mp"),
                    "recall": metric_value(metrics, "mr"),
                    "mAP50": metric_value(metrics, "map50"),
                    "mAP50-95": metric_value(metrics, "map"),
                    "run_dir": str(args.project / run_name),
                }
            )

    summary_path = root / "experiments_sci4" / "subset_validation_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote subset validation summary: {summary_path}")


if __name__ == "__main__":
    main()
