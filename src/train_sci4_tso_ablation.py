from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def set_local_ultralytics_config(root: Path) -> None:
    cfg = root / ".ultralytics"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))


def read_plan(plan_path: Path) -> list[dict[str, str]]:
    with plan_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: str) -> float:
    return float(str(value).strip())


def as_int(value: str) -> int:
    return int(float(str(value).strip()))


def train_one(root: Path, row: dict[str, str], data: Path, project: Path, device: str, workers: int, batch_override: int | None, amp: bool) -> None:
    from ultralytics import YOLO

    model_yaml = row["model_yaml"]
    if model_yaml == "src/yolov8m.yaml_or_pretrained_yolov8m.pt":
        model_path = root / "src" / "yolov8m.pt"
    else:
        model_path = root / model_yaml

    model = YOLO(str(model_path))
    pretrained = root / row["pretrained"]
    if model_path.suffix.lower() in {".yaml", ".yml"} and pretrained.exists():
        model.load(str(pretrained))

    batch = batch_override if batch_override is not None else as_int(row["batch"])
    name = row["output_name"]

    print("=" * 100)
    print(f"Training {name}")
    print(f"Model: {model_path}")
    print(f"Data: {data}")
    print(f"Project: {project}")
    print("=" * 100)

    model.train(
        data=str(data),
        epochs=as_int(row["epochs"]),
        imgsz=as_int(row["imgsz"]),
        batch=batch,
        workers=workers,
        device=device,
        optimizer=row["optimizer"],
        lr0=as_float(row["lr0"]),
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=as_float(row["warmup_epochs"]),
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=as_bool(row["cos_lr"]),
        patience=60,
        mosaic=as_float(row["mosaic"]),
        close_mosaic=as_int(row["close_mosaic"]),
        mixup=as_float(row["mixup"]),
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        rect=False,
        multi_scale=False,
        amp=amp,
        save=True,
        save_period=10,
        plots=True,
        project=str(project),
        name=name,
        exist_ok=False,
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCI4 TSO/BiFPN ablation experiments.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=None, help="Override plan batch size, useful for small GPUs.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP and its preflight check.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected experiment names from the plan.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    set_local_ultralytics_config(root)

    plan_path = args.plan or (root / "experiments_sci4" / "tso_ablation_plan.csv")
    data = (args.data or (root / "dataset" / "final.yaml")).resolve()
    project = (args.project or (root / "runs" / "detect_sci4")).resolve()
    project.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or [])
    rows = read_plan(plan_path)
    if selected:
        rows = [row for row in rows if row["experiment"] in selected or row["output_name"] in selected]

    if not rows:
        raise SystemExit("No experiments selected.")

    for row in rows:
        out_dir = project / row["output_name"]
        if args.skip_existing and (out_dir / "results.csv").exists():
            print(f"Skipping existing run: {out_dir}")
            continue
        train_one(root, row, data, project, args.device, args.workers, args.batch, not args.no_amp)


if __name__ == "__main__":
    main()
