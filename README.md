# YOLOv8m-BiFPN-MixUp: UAV Object Detection with BiFPN and MixUp Training Strategy Selection

Code, model configurations, and result evidence for the paper:

> **A YOLOv8m-Based UAV Object Detection Method with BiFPN and MixUp Training Strategy Selection**
> Zihao Zhang, Pengyuan Wang, Huiling Huang, Yan Bai (corresponding author)
> College of Control and Information Engineering, Northeast Forestry University
> Submitted to the *Journal of Electronic Imaging* (SPIE).

The repository contains everything needed to re-run the training strategies and
to inspect the quantitative evidence reported in the paper. It intentionally
does **not** contain the datasets (both are public; see
[DATASETS.md](DATASETS.md)) or the trained weights (available from the
corresponding author on reasonable request, as stated in the paper's Data
Availability statement).

## Method summary

1. **BiFPN Neck** — the PAN-FPN neck of YOLOv8m is replaced by a four-node
   bidirectional feature pyramid with learnable weighted fusion
   (`BiFPN_Concat2` nodes, see `src/yolov8m-bifpn.yaml`).
2. **Training-strategy selection** — SGD, Warmup, Close Mosaic, and MixUp are
   treated as separate candidate components and combined through a
   component-level ablation matrix (`experiments_sci4/tso_ablation_plan.csv`,
   driven by `src/train_sci4_tso_ablation.py`). MixUp is the component most
   compatible with BiFPN, giving the final **YOLOv8m-BiFPN-MixUp** model.
3. **Validation** — main experiments on VisDrone2019-DET, difficult-subset
   validation (tiny / small-heavy / occlusion / severe-occlusion / truncation /
   crowded, top-100 images each), cross-scale (v8s/m/l) and cross-dataset
   (UAVDT) generalization, and comparisons against CBAM / ECA / C3TR /
   YOLO11m / RT-DETR-l.

## Headline results (VisDrone2019-DET, val, imgsz=1024)

| Model | mAP50 | mAP50-95 |
| --- | --- | --- |
| YOLOv8m (baseline) | 52.28 | 32.24 |
| **YOLOv8m-BiFPN-MixUp (ours)** | **54.60** | **33.94** |

- Difficult-subset average mAP50: 50.50 → 52.61
- UAVDT (cross-dataset): mAP50-95 60.22 → 61.36
- Efficiency (val @1024, RTX 4090D): 77.6 FPS for the proposed model, see
  `experiments_sci4/efficiency_compare/final_efficiency_comparison_val1024.csv`.

All intermediate ablation numbers are in
`experiments_sci4/all_ablation_summary_best.csv`.

## Repository layout

```
├── README.md                     this file
├── DATASETS.md                   dataset download + preparation instructions
├── requirements.txt
├── dataset/                      data YAML templates (edit `path:` before use)
│   ├── final.yaml                VisDrone2019 (default `--data` of the ablation script)
│   └── uavdt.yaml
├── src/                          runnable entry points (relative paths, `--root` aware)
│   ├── train_sci4_tso_ablation.py     component-level ablation / main training driver
│   ├── validate_sci4_subsets.py       difficult-subset validation
│   ├── prepare_sci4_experiments.py    per-image difficulty statistics + subset lists
│   ├── reconvert_visdrone_unified.py  VisDrone -> YOLO label conversion
│   ├── yolov8{m,s,l}-bifpn.yaml       BiFPN neck model definitions
│   ├── attention_compare/             CBAM / ECA / C3TR comparison model YAMLs
│   └── cloud_provenance/              original scripts run on the cloud instance
│                                      (kept verbatim with /root paths, for provenance)
├── experiments_sci4/             quantitative evidence (CSV / PNG / PDF / MD)
└── ultralytics_patch/            custom-module patch for ultralytics 8.4.37
```

## Environment

- Python 3.12, PyTorch 2.8.0 + CUDA 12.8 (any recent CUDA build works)
- `ultralytics==8.4.37` (pinned; the patch targets this version)
- 1x GPU with >= 16 GB memory for the paper configuration
  (`imgsz=1024, batch=8`; a 24 GB RTX 4090D was used)
- see `requirements.txt`

```bash
pip install -r requirements.txt
python ultralytics_patch/apply_patch.py   # install BiFPN / attention modules
```

## Datasets

Both datasets are public. Download and preparation steps (label conversion,
directory layout, YAML edits) are documented in [DATASETS.md](DATASETS.md).

- VisDrone2019-DET — <https://github.com/VisDrone/VisDrone-Dataset>
- UAVDT — <https://sites.google.com/site/daviddo0323/projects/uavdt>

Expected layout after preparation (paths configured in `dataset/*.yaml`):

```
dataset/
├── VisDrone2019-DET-train/{images,annotations,labels}
├── VisDrone2019-DET-val/{images,annotations,labels}
```

## Reproducing the paper

Run from the repository root. Pretrained weights referenced below are
downloaded into `src/` (e.g. `yolov8m.pt` from the
[Ultralytics releases](https://github.com/ultralytics/assets/releases)).

**1. Component-level ablation (Table "ablation", A1–A4 / B1–B4) and the
final BiFPN + MixUp model:**

```bash
python src/train_sci4_tso_ablation.py --only \
    A1_sgd_only A2_warmup_only A3_close_mosaic_only A4_mixup_only \
    A7_tso_all_no_bifpn B1_bifpn_sgd B2_bifpn_warmup B3_bifpn_close_mosaic \
    B4_bifpn_mixup --device 0 --workers 8
```

The experiment matrix (optimizer / warmup / close-mosaic / mixup / BiFPN per
run, 80 epochs, imgsz 1024, batch 8) is defined in
`experiments_sci4/tso_ablation_plan.csv`; run the full matrix with
`--skip-existing` instead of `--only`. Results land in `runs/detect_sci4/`.

**2. Difficult-subset validation:**

```bash
python src/validate_sci4_subsets.py --device 0 --workers 8
```

Subset lists (top-100 hardest val images per attribute, produced by
`prepare_sci4_experiments.py` from the per-image statistics in
`experiments_sci4/val_image_difficulty.csv`) are in
`experiments_sci4/val_subsets/`.

**3. Cross-scale / cross-dataset generalization and comparisons** —
`src/cloud_provenance/` keeps the exact scripts used on the training instance
(YOLOv8s/v8l BiFPN runs, CBAM/ECA/C3TR, YOLO11m, RT-DETR-l, efficiency table,
UAVDT). They contain hard-coded `/root/...` instance paths and are provided
as-is for provenance; adjust the paths to your machine if you re-run them.

**4. BiFPN fusion-weight analysis** — learned weights are read from the
trained checkpoints; the extracted values and plots are in
`experiments_sci4/bifpn_weight_analysis/`.

## Evidence index (`experiments_sci4/`)

| File | Content |
| --- | --- |
| `tso_ablation_plan.csv` / `.json` | full experiment matrix of the training-strategy selection |
| `all_ablation_summary_best.csv` | best-epoch P/R/mAP50/mAP50-95 for every ablation run |
| `difficulty_subset_validation_final.csv` | baseline vs. ours on the six difficult subsets (`YOLOv8m_baseline` = pretrained-YOLOv8m run, `Old_BiFPN_TSO` = 150-epoch BiFPN+TSO run, `B4_BiFPN_MixUp` = final model) |
| `val_image_difficulty.csv`, `val_subsets/`, `val_subsets_summary.csv` | per-image difficulty statistics and the top-100 subset lists |
| `bifpn_generalization_summary.csv` | BiFPN / BiFPN-TSO deltas at YOLOv8-s/m/l scale |
| `uavdt/uavdt.yaml`, `uavdt/uavdt_generalization_summary.csv` | UAVDT configuration and the extracted per-epoch best metrics of the two generalization runs (baseline 0.602 vs. ours 0.614 mAP50-95, as logged per epoch with 3-decimal rounding; paper reports 60.22 / 61.36 from the full-precision results.csv) |
| `efficiency_compare/` | params, GFLOPs, per-stage latency, FPS @ val 1024 for all compared methods |
| `attention_compare/` | CBAM / ECA / C3TR comparison summaries |
| `bifpn_weight_analysis/` | learned fusion-weight tables, bar chart, direction-contribution figure |

## Data availability

The trained weights, complete training logs, and full run outputs are not
included in this repository due to size; they are available from the
corresponding author (baiyan@nefu.edu.cn) on reasonable request.
