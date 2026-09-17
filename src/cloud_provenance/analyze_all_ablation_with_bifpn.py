import pandas as pd
from pathlib import Path

RUNS = {
    "YOLOv8m Baseline": "/root/runs/detect/visdrone_verify_80ep2",
    "YOLOv8m-P2-80ep": "/root/runs/detect/visdrone_yolov8m_p2_80ep",
    "YOLOv8m-P2-fixed-150ep": "/root/runs/detect/visdrone_yolov8m_p2_fixed_150ep",
    "YOLOv8m-SPDConv-80ep": "/root/runs/detect/visdrone_yolov8m_spd_80ep",
    "YOLOv8m-EMA-80ep": "/root/runs/detect/visdrone_yolov8m_ema_80ep",
    "YOLOv8m-BiFPN-80ep": "/root/runs/detect/visdrone_yolov8m_bifpn_80ep",
}

def load_result(name, run_path):
    run_path = Path(run_path)
    csv_path = run_path / "results.csv"

    if not csv_path.exists():
        print(f"[跳过] {name}: 找不到 {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    best50 = df.loc[df["metrics/mAP50(B)"].idxmax()]
    best5095 = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    last = df.iloc[-1]

    return {
        "name": name,
        "path": run_path,

        "last_epoch": int(last["epoch"]),
        "last_p": float(last["metrics/precision(B)"]),
        "last_r": float(last["metrics/recall(B)"]),
        "last_map50": float(last["metrics/mAP50(B)"]),
        "last_map5095": float(last["metrics/mAP50-95(B)"]),

        "best50_epoch": int(best50["epoch"]),
        "best50_p": float(best50["metrics/precision(B)"]),
        "best50_r": float(best50["metrics/recall(B)"]),
        "best50_map50": float(best50["metrics/mAP50(B)"]),
        "best50_map5095": float(best50["metrics/mAP50-95(B)"]),

        "best5095_epoch": int(best5095["epoch"]),
        "best5095_p": float(best5095["metrics/precision(B)"]),
        "best5095_r": float(best5095["metrics/recall(B)"]),
        "best5095_map50": float(best5095["metrics/mAP50(B)"]),
        "best5095_map5095": float(best5095["metrics/mAP50-95(B)"]),

        "train_box": float(last["train/box_loss"]),
        "train_cls": float(last["train/cls_loss"]),
        "train_dfl": float(last["train/dfl_loss"]),
        "val_box": float(last["val/box_loss"]),
        "val_cls": float(last["val/cls_loss"]),
        "val_dfl": float(last["val/dfl_loss"]),
    }

results = []
for name, path in RUNS.items():
    r = load_result(name, path)
    if r:
        results.append(r)

baseline = next((r for r in results if r["name"] == "YOLOv8m Baseline"), None)

print("\n" + "=" * 120)
print("一、按最佳 mAP50 对比")
print("=" * 120)
print(f"{'Model':<34}{'BestEp':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 95)

for r in results:
    print(
        f"{r['name']:<34}"
        f"{r['best50_epoch']:>8}"
        f"{r['best50_p']:>10.4f}"
        f"{r['best50_r']:>10.4f}"
        f"{r['best50_map50']:>10.4f}"
        f"{r['best50_map5095']:>12.4f}"
    )

print("\n" + "=" * 120)
print("二、按最佳 mAP50-95 对比")
print("=" * 120)
print(f"{'Model':<34}{'BestEp':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 95)

for r in results:
    print(
        f"{r['name']:<34}"
        f"{r['best5095_epoch']:>8}"
        f"{r['best5095_p']:>10.4f}"
        f"{r['best5095_r']:>10.4f}"
        f"{r['best5095_map50']:>10.4f}"
        f"{r['best5095_map5095']:>12.4f}"
    )

print("\n" + "=" * 120)
print("三、相对 Baseline 提升/下降")
print("=" * 120)
print(f"{'Model':<34}{'ΔP':>10}{'ΔR':>10}{'ΔmAP50':>12}{'ΔmAP50-95':>14}")
print("-" * 95)

if baseline:
    for r in results:
        if r["name"] == "YOLOv8m Baseline":
            continue

        print(
            f"{r['name']:<34}"
            f"{r['best50_p'] - baseline['best50_p']:>10.4f}"
            f"{r['best50_r'] - baseline['best50_r']:>10.4f}"
            f"{r['best50_map50'] - baseline['best50_map50']:>12.4f}"
            f"{r['best50_map5095'] - baseline['best50_map5095']:>14.4f}"
        )

print("\n" + "=" * 120)
print("四、论文消融实验表")
print("=" * 120)
print(f"{'Model':<34}{'P2':>6}{'SPD':>6}{'EMA':>6}{'BiFPN':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 115)

for r in results:
    p2 = "√" if "P2" in r["name"] else "×"
    spd = "√" if "SPDConv" in r["name"] else "×"
    ema = "√" if "EMA" in r["name"] else "×"
    bifpn = "√" if "BiFPN" in r["name"] else "×"

    print(
        f"{r['name']:<34}"
        f"{p2:>6}"
        f"{spd:>6}"
        f"{ema:>6}"
        f"{bifpn:>8}"
        f"{r['best50_p']:>10.4f}"
        f"{r['best50_r']:>10.4f}"
        f"{r['best50_map50']:>10.4f}"
        f"{r['best50_map5095']:>12.4f}"
    )

print("\n" + "=" * 120)
print("五、最后一轮 Loss 对比")
print("=" * 120)
print(f"{'Model':<34}{'train_box':>12}{'train_cls':>12}{'train_dfl':>12}{'val_box':>12}{'val_cls':>12}{'val_dfl':>12}")
print("-" * 115)

for r in results:
    print(
        f"{r['name']:<34}"
        f"{r['train_box']:>12.4f}"
        f"{r['train_cls']:>12.4f}"
        f"{r['train_dfl']:>12.4f}"
        f"{r['val_box']:>12.4f}"
        f"{r['val_cls']:>12.4f}"
        f"{r['val_dfl']:>12.4f}"
    )

print("\n" + "=" * 120)
print("六、结果文件路径")
print("=" * 120)

for r in results:
    print(f"\n{r['name']}:")
    for f in [
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "weights/best.pt",
        "weights/last.pt",
    ]:
        fp = r["path"] / f
        print(f"  {fp} {'存在' if fp.exists() else '不存在'}")

print("\n分析完成。")
