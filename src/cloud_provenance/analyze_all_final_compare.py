import pandas as pd
from pathlib import Path

RUNS = {
    "YOLOv5s": "/root/runs/detect/compare_yolov5s_80ep",

    "YOLOv8s": "/root/runs/detect/compare_yolov8s_80ep",
    "YOLOv8s-BiFPN": "/root/runs/detect/compare_yolov8s_bifpn_80ep",
    "YOLOv8s-BiFPN-Opt": "/root/runs/detect/compare_yolov8s_bifpn_opt_150ep",

    "YOLOv8m": "/root/runs/detect/visdrone_verify_80ep2",
    "YOLOv8m-BiFPN": "/root/runs/detect/visdrone_yolov8m_bifpn_80ep",
    "YOLOv8m-BiFPN-Opt": "/root/runs/detect/final_yolov8m_bifpn_opt_150ep",

    "YOLOv8l": "/root/runs/detect/compare_yolov8l_80ep",
    "YOLOv8l-BiFPN": "/root/runs/detect/compare_yolov8l_bifpn_80ep",
    "YOLOv8l-BiFPN-Opt": "/root/runs/detect/compare_yolov8l_bifpn_opt_150ep",

    "YOLOv10s": "/root/runs/detect/compare_yolov10s_80ep",
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
    best95 = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    last = df.iloc[-1]

    return {
        "Model": name,
        "Path": str(path),

        "Best_mAP50_Epoch": int(best50["epoch"]),
        "P": float(best50["metrics/precision(B)"]),
        "R": float(best50["metrics/recall(B)"]),
        "mAP50": float(best50["metrics/mAP50(B)"]),
        "mAP50-95": float(best50["metrics/mAP50-95(B)"]),

        "Best_mAP50-95_Epoch": int(best95["epoch"]),
        "P_at_best95": float(best95["metrics/precision(B)"]),
        "R_at_best95": float(best95["metrics/recall(B)"]),
        "mAP50_at_best95": float(best95["metrics/mAP50(B)"]),
        "Best_mAP50-95": float(best95["metrics/mAP50-95(B)"]),

        "Last_Epoch": int(last["epoch"]),
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

df = pd.DataFrame(results)

print("\n" + "=" * 130)
print("一、全部模型总对比：按最佳 mAP50")
print("=" * 130)
print(df[[
    "Model", "Best_mAP50_Epoch", "P", "R", "mAP50", "mAP50-95"
]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 130)
print("二、全部模型总对比：按最佳 mAP50-95")
print("=" * 130)
print(df[[
    "Model", "Best_mAP50-95_Epoch", "P_at_best95", "R_at_best95", "mAP50_at_best95", "Best_mAP50-95"
]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 130)
print("三、YOLOv8s / YOLOv8m / YOLOv8l 泛化实验提升")
print("=" * 130)

pairs = [
    ("YOLOv8s", "YOLOv8s-BiFPN", "YOLOv8s-BiFPN-Opt"),
    ("YOLOv8m", "YOLOv8m-BiFPN", "YOLOv8m-BiFPN-Opt"),
    ("YOLOv8l", "YOLOv8l-BiFPN", "YOLOv8l-BiFPN-Opt"),
]

for base, bifpn, opt in pairs:
    if base not in df["Model"].values:
        continue

    b = df[df["Model"] == base].iloc[0]
    print("\n" + "-" * 100)
    print(f"{base} 系列")
    print("-" * 100)
    print(f"{'Model':<28}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}{'ΔmAP50':>12}{'ΔmAP50-95':>14}")

    for name in [base, bifpn, opt]:
        if name not in df["Model"].values:
            continue
        r = df[df["Model"] == name].iloc[0]
        print(
            f"{name:<28}"
            f"{r['P']:>10.4f}"
            f"{r['R']:>10.4f}"
            f"{r['mAP50']:>10.4f}"
            f"{r['mAP50-95']:>12.4f}"
            f"{r['mAP50'] - b['mAP50']:>12.4f}"
            f"{r['mAP50-95'] - b['mAP50-95']:>14.4f}"
        )

print("\n" + "=" * 130)
print("四、BiFPN+Opt 相对 Baseline 提升汇总")
print("=" * 130)
print(f"{'Scale':<8}{'Baseline':<20}{'Ours':<26}{'ΔP':>10}{'ΔR':>10}{'ΔmAP50':>12}{'ΔmAP50-95':>14}{'Rel_mAP50':>12}")

for scale, base, opt in [
    ("s", "YOLOv8s", "YOLOv8s-BiFPN-Opt"),
    ("m", "YOLOv8m", "YOLOv8m-BiFPN-Opt"),
    ("l", "YOLOv8l", "YOLOv8l-BiFPN-Opt"),
]:
    if base not in df["Model"].values or opt not in df["Model"].values:
        continue

    b = df[df["Model"] == base].iloc[0]
    o = df[df["Model"] == opt].iloc[0]

    print(
        f"{scale:<8}"
        f"{base:<20}"
        f"{opt:<26}"
        f"{o['P'] - b['P']:>10.4f}"
        f"{o['R'] - b['R']:>10.4f}"
        f"{o['mAP50'] - b['mAP50']:>12.4f}"
        f"{o['mAP50-95'] - b['mAP50-95']:>14.4f}"
        f"{(o['mAP50'] - b['mAP50']) / b['mAP50'] * 100:>11.2f}%"
    )

print("\n" + "=" * 130)
print("五、主流算法对比表")
print("=" * 130)

main_models = [
    "YOLOv5s",
    "YOLOv8s",
    "YOLOv8m",
    "YOLOv8l",
    "YOLOv10s",
    "YOLOv8m-BiFPN-Opt",
    "YOLOv8l-BiFPN-Opt",
]

main_df = df[df["Model"].isin(main_models)][[
    "Model", "P", "R", "mAP50", "mAP50-95"
]]

print(main_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 130)
print("六、最后一轮 Loss 对比")
print("=" * 130)
print(df[[
    "Model", "Last_Epoch", "train_box", "train_cls", "train_dfl", "val_box", "val_cls", "val_dfl"
]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 130)
print("七、论文结论自动生成")
print("=" * 130)

for scale, base, opt in [
    ("YOLOv8s", "YOLOv8s", "YOLOv8s-BiFPN-Opt"),
    ("YOLOv8m", "YOLOv8m", "YOLOv8m-BiFPN-Opt"),
    ("YOLOv8l", "YOLOv8l", "YOLOv8l-BiFPN-Opt"),
]:
    if base in df["Model"].values and opt in df["Model"].values:
        b = df[df["Model"] == base].iloc[0]
        o = df[df["Model"] == opt].iloc[0]
        print(f"\n{scale}:")
        print(f"Baseline: mAP50={b['mAP50']:.4f}, mAP50-95={b['mAP50-95']:.4f}")
        print(f"BiFPN+Opt: mAP50={o['mAP50']:.4f}, mAP50-95={o['mAP50-95']:.4f}")
        print(f"提升: mAP50 {o['mAP50'] - b['mAP50']:+.4f}, mAP50-95 {o['mAP50-95'] - b['mAP50-95']:+.4f}")

best = df.loc[df["mAP50"].idxmax()]
print("\n最佳 mAP50 模型:")
print(f"{best['Model']} | mAP50={best['mAP50']:.4f} | mAP50-95={best['mAP50-95']:.4f}")

best95 = df.loc[df["mAP50-95"].idxmax()]
print("\n最佳 mAP50-95 模型:")
print(f"{best95['Model']} | mAP50={best95['mAP50']:.4f} | mAP50-95={best95['mAP50-95']:.4f}")

print("\n" + "=" * 130)
print("八、结果文件路径")
print("=" * 130)

for _, r in df.iterrows():
    p = Path(r["Path"])
    print(f"\n{r['Model']}: {p}")
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
    ]:
        fp = p / f
        print(f"  {fp} {'存在' if fp.exists() else '不存在'}")

out_csv = "/root/src/final_all_compare_summary.csv"
df.to_csv(out_csv, index=False)
print("\n" + "=" * 130)
print(f"CSV汇总表已保存: {out_csv}")
print("分析完成。")
