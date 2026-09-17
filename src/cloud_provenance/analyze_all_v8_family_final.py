import pandas as pd
from pathlib import Path

RUNS = {
    # YOLOv8s
    "YOLOv8s Baseline": "/root/runs/detect/compare_yolov8s_80ep",
    "YOLOv8s-BiFPN": "/root/runs/detect/compare_yolov8s_bifpn_80ep",
    "YOLOv8s-BiFPN-Opt": "/root/runs/detect/compare_yolov8s_bifpn_opt_150ep",

    # YOLOv8m
    "YOLOv8m Baseline": "/root/runs/detect/visdrone_verify_80ep2",
    "YOLOv8m-BiFPN": "/root/runs/detect/visdrone_yolov8m_bifpn_80ep",
    "YOLOv8m-BiFPN-Opt": "/root/runs/detect/final_yolov8m_bifpn_opt_150ep",

    # YOLOv8l
    "YOLOv8l Baseline": "/root/runs/detect/compare_yolov8l_80ep",
    "YOLOv8l-BiFPN": "/root/runs/detect/compare_yolov8l_bifpn_80ep",
    "YOLOv8l-BiFPN-Opt": "/root/runs/detect/compare_yolov8l_bifpn_opt_150ep",
}

def load_result(name, path):
    path = Path(path)
    csv = path / "results.csv"

    if not csv.exists():
        print(f"[跳过] {name}: 找不到 {csv}")
        return None

    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]

    best50 = df.loc[df["metrics/mAP50(B)"].idxmax()]
    best5095 = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    last = df.iloc[-1]

    return {
        "name": name,
        "path": path,

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

        "last_epoch": int(last["epoch"]),
        "last_p": float(last["metrics/precision(B)"]),
        "last_r": float(last["metrics/recall(B)"]),
        "last_map50": float(last["metrics/mAP50(B)"]),
        "last_map5095": float(last["metrics/mAP50-95(B)"]),

        "train_box": float(last["train/box_loss"]),
        "train_cls": float(last["train/cls_loss"]),
        "train_dfl": float(last["train/dfl_loss"]),
        "val_box": float(last["val/box_loss"]),
        "val_cls": float(last["val/cls_loss"]),
        "val_dfl": float(last["val/dfl_loss"]),
    }

results = {}
for name, path in RUNS.items():
    r = load_result(name, path)
    if r:
        results[name] = r

print("\n" + "=" * 125)
print("一、YOLOv8 系列全部实验结果：按最佳 mAP50")
print("=" * 125)
print(f"{'Model':<30}{'BestEp':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 90)

for name in RUNS:
    if name in results:
        r = results[name]
        print(
            f"{name:<30}"
            f"{r['best50_epoch']:>8}"
            f"{r['best50_p']:>10.4f}"
            f"{r['best50_r']:>10.4f}"
            f"{r['best50_map50']:>10.4f}"
            f"{r['best50_map5095']:>12.4f}"
        )

print("\n" + "=" * 125)
print("二、YOLOv8 系列全部实验结果：按最佳 mAP50-95")
print("=" * 125)
print(f"{'Model':<30}{'BestEp':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 90)

for name in RUNS:
    if name in results:
        r = results[name]
        print(
            f"{name:<30}"
            f"{r['best5095_epoch']:>8}"
            f"{r['best5095_p']:>10.4f}"
            f"{r['best5095_r']:>10.4f}"
            f"{r['best5095_map50']:>10.4f}"
            f"{r['best5095_map5095']:>12.4f}"
        )

def compare(base, bifpn, opt, title):
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)

    if base not in results:
        print(f"缺少 {base}")
        return

    b = results[base]

    print(f"{'Model':<30}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}{'ΔmAP50':>12}{'ΔmAP50-95':>14}")
    print("-" * 105)

    for name in [base, bifpn, opt]:
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:<30}"
            f"{r['best50_p']:>10.4f}"
            f"{r['best50_r']:>10.4f}"
            f"{r['best50_map50']:>10.4f}"
            f"{r['best50_map5095']:>12.4f}"
            f"{r['best50_map50'] - b['best50_map50']:>12.4f}"
            f"{r['best50_map5095'] - b['best50_map5095']:>14.4f}"
        )

compare(
    "YOLOv8s Baseline",
    "YOLOv8s-BiFPN",
    "YOLOv8s-BiFPN-Opt",
    "三、YOLOv8s 泛化实验对比"
)

compare(
    "YOLOv8m Baseline",
    "YOLOv8m-BiFPN",
    "YOLOv8m-BiFPN-Opt",
    "四、YOLOv8m 主实验与消融对比"
)

compare(
    "YOLOv8l Baseline",
    "YOLOv8l-BiFPN",
    "YOLOv8l-BiFPN-Opt",
    "五、YOLOv8l 泛化实验对比"
)

print("\n" + "=" * 125)
print("六、BiFPN+Opt 相对 Baseline 提升汇总")
print("=" * 125)
print(f"{'Scale':<10}{'Baseline':<24}{'BiFPN+Opt':<28}{'ΔP':>10}{'ΔR':>10}{'ΔmAP50':>12}{'ΔmAP50-95':>14}{'Rel mAP50':>12}")
print("-" * 120)

pairs = [
    ("s", "YOLOv8s Baseline", "YOLOv8s-BiFPN-Opt"),
    ("m", "YOLOv8m Baseline", "YOLOv8m-BiFPN-Opt"),
    ("l", "YOLOv8l Baseline", "YOLOv8l-BiFPN-Opt"),
]

for scale, base, opt in pairs:
    if base not in results or opt not in results:
        continue

    b = results[base]
    o = results[opt]

    dp = o["best50_p"] - b["best50_p"]
    dr = o["best50_r"] - b["best50_r"]
    dm50 = o["best50_map50"] - b["best50_map50"]
    dm95 = o["best50_map5095"] - b["best50_map5095"]
    rel = dm50 / b["best50_map50"] * 100 if b["best50_map50"] else 0

    print(
        f"{scale:<10}"
        f"{base:<24}"
        f"{opt:<28}"
        f"{dp:>10.4f}"
        f"{dr:>10.4f}"
        f"{dm50:>12.4f}"
        f"{dm95:>14.4f}"
        f"{rel:>11.2f}%"
    )

print("\n" + "=" * 125)
print("七、论文泛化实验表")
print("=" * 125)
print(f"{'Model':<28}{'BiFPN':>8}{'Opt':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 95)

for name in RUNS:
    if name not in results:
        continue

    r = results[name]
    bifpn = "√" if "BiFPN" in name else "×"
    opt = "√" if "Opt" in name else "×"

    print(
        f"{name:<28}"
        f"{bifpn:>8}"
        f"{opt:>8}"
        f"{r['best50_p']:>10.4f}"
        f"{r['best50_r']:>10.4f}"
        f"{r['best50_map50']:>10.4f}"
        f"{r['best50_map5095']:>12.4f}"
    )

print("\n" + "=" * 125)
print("八、最后一轮 Loss 对比")
print("=" * 125)
print(f"{'Model':<30}{'train_box':>12}{'train_cls':>12}{'train_dfl':>12}{'val_box':>12}{'val_cls':>12}{'val_dfl':>12}")
print("-" * 105)

for name in RUNS:
    if name not in results:
        continue

    r = results[name]
    print(
        f"{name:<30}"
        f"{r['train_box']:>12.4f}"
        f"{r['train_cls']:>12.4f}"
        f"{r['train_dfl']:>12.4f}"
        f"{r['val_box']:>12.4f}"
        f"{r['val_cls']:>12.4f}"
        f"{r['val_dfl']:>12.4f}"
    )

print("\n" + "=" * 125)
print("九、论文结论自动生成")
print("=" * 125)

for scale, base, opt in pairs:
    if base not in results or opt not in results:
        continue

    b = results[base]
    o = results[opt]

    dm50 = o["best50_map50"] - b["best50_map50"]
    dm95 = o["best50_map5095"] - b["best50_map5095"]

    print(f"\nYOLOv8{scale}:")
    print(f"Baseline mAP50={b['best50_map50']:.4f}, mAP50-95={b['best50_map5095']:.4f}")
    print(f"BiFPN+Opt mAP50={o['best50_map50']:.4f}, mAP50-95={o['best50_map5095']:.4f}")
    print(f"提升：mAP50 {dm50:+.4f}, mAP50-95 {dm95:+.4f}")

    if dm50 > 0 and dm95 > 0:
        print("结论：该规模模型上 BiFPN+Opt 取得有效提升。")
    elif dm50 > 0:
        print("结论：该规模模型上 mAP50 有提升，但 mAP50-95 提升有限。")
    else:
        print("结论：该规模模型上提升不明显，需要结合复杂度分析。")

print("\n" + "=" * 125)
print("十、结果文件路径")
print("=" * 125)

for name in RUNS:
    if name not in results:
        continue

    p = results[name]["path"]
    print(f"\n{name}: {p}")
    for f in [
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "weights/best.pt",
        "weights/last.pt",
    ]:
        fp = p / f
        print(f"  {fp} {'存在' if fp.exists() else '不存在'}")

print("\n分析完成。")
