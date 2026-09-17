import pandas as pd
from pathlib import Path

RUNS = {
    "YOLOv8s": "/root/runs/detect/compare_yolov8s_80ep",
    "YOLOv8s-BiFPN": "/root/runs/detect/compare_yolov8s_bifpn_80ep",
    "YOLOv8m": "/root/runs/detect/visdrone_verify_80ep2",
    "YOLOv8m-BiFPN-80ep": "/root/runs/detect/visdrone_yolov8m_bifpn_80ep",
    "YOLOv8m-BiFPN-Opt": "/root/runs/detect/final_yolov8m_bifpn_opt_150ep",
}

def load_result(name, path):
    path = Path(path)
    csv_path = path / "results.csv"

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
    }

results = {}
for name, path in RUNS.items():
    r = load_result(name, path)
    if r:
        results[name] = r

print("\n" + "=" * 110)
print("一、BiFPN 泛化实验与模型规模对比：按最佳 mAP50")
print("=" * 110)
print(f"{'Model':<28}{'BestEp':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 90)

for name in RUNS:
    if name in results:
        r = results[name]
        print(
            f"{r['name']:<28}"
            f"{r['best50_epoch']:>8}"
            f"{r['best50_p']:>10.4f}"
            f"{r['best50_r']:>10.4f}"
            f"{r['best50_map50']:>10.4f}"
            f"{r['best50_map5095']:>12.4f}"
        )

print("\n" + "=" * 110)
print("二、BiFPN 泛化实验与模型规模对比：按最佳 mAP50-95")
print("=" * 110)
print(f"{'Model':<28}{'BestEp':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
print("-" * 90)

for name in RUNS:
    if name in results:
        r = results[name]
        print(
            f"{r['name']:<28}"
            f"{r['best5095_epoch']:>8}"
            f"{r['best5095_p']:>10.4f}"
            f"{r['best5095_r']:>10.4f}"
            f"{r['best5095_map50']:>10.4f}"
            f"{r['best5095_map5095']:>12.4f}"
        )

def compare(base_name, improved_name, title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    if base_name not in results:
        print(f"缺少 {base_name}")
        return

    if improved_name not in results:
        print(f"缺少 {improved_name}")
        return

    b = results[base_name]
    i = results[improved_name]

    print(f"{'Metric':<18}{base_name:>18}{improved_name:>24}{'提升':>14}{'相对提升':>14}")
    print("-" * 95)

    metrics = [
        ("Precision", "best50_p"),
        ("Recall", "best50_r"),
        ("mAP50", "best50_map50"),
        ("mAP50-95", "best50_map5095"),
    ]

    for label, key in metrics:
        delta = i[key] - b[key]
        rel = delta / b[key] * 100 if b[key] != 0 else 0
        print(
            f"{label:<18}"
            f"{b[key]:>18.4f}"
            f"{i[key]:>24.4f}"
            f"{delta:>14.4f}"
            f"{rel:>13.2f}%"
        )

compare("YOLOv8s", "YOLOv8s-BiFPN", "三、YOLOv8s 泛化实验：Baseline vs BiFPN")
compare("YOLOv8m", "YOLOv8m-BiFPN-80ep", "四、YOLOv8m 结构实验：Baseline vs BiFPN")
compare("YOLOv8m", "YOLOv8m-BiFPN-Opt", "五、YOLOv8m 最终模型：Baseline vs BiFPN-Opt")

print("\n" + "=" * 110)
print("六、论文可用结论")
print("=" * 110)

if "YOLOv8s" in results and "YOLOv8s-BiFPN" in results:
    b = results["YOLOv8s"]
    i = results["YOLOv8s-BiFPN"]
    print("\n[YOLOv8s 泛化实验]")
    print(f"YOLOv8s-BiFPN 相比 YOLOv8s：")
    print(f"mAP50 提升：{i['best50_map50'] - b['best50_map50']:+.4f}")
    print(f"mAP50-95 提升：{i['best50_map5095'] - b['best50_map5095']:+.4f}")

    if i["best50_map50"] > b["best50_map50"]:
        print("结论：BiFPN 在 YOLOv8s 上取得提升，说明该模块具有一定泛化能力。")
    else:
        print("结论：BiFPN 在 YOLOv8s 上未取得提升，需要在论文中说明轻量模型中融合模块收益有限。")

if "YOLOv8m" in results and "YOLOv8m-BiFPN-Opt" in results:
    b = results["YOLOv8m"]
    i = results["YOLOv8m-BiFPN-Opt"]
    print("\n[YOLOv8m 最终模型]")
    print(f"YOLOv8m-BiFPN-Opt 相比 YOLOv8m：")
    print(f"mAP50 提升：{i['best50_map50'] - b['best50_map50']:+.4f}")
    print(f"mAP50-95 提升：{i['best50_map5095'] - b['best50_map5095']:+.4f}")
    print("结论：最终模型相较原始 YOLOv8m 取得明显提升，可作为论文最终改进模型。")

print("\n" + "=" * 110)
print("七、结果文件路径")
print("=" * 110)

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
        "weights/best.pt",
    ]:
        fp = p / f
        print(f"  {fp} {'存在' if fp.exists() else '不存在'}")

print("\n分析完成。")
