# Datasets

Both datasets used in the paper are publicly available. This repository ships
only the YAML templates (`dataset/final.yaml`, `dataset/uavdt.yaml`); edit the
`path:` field in each template to point at your local copy.

## VisDrone2019-DET (main experiments)

- Download: <https://github.com/VisDrone/VisDrone-Dataset>
- Files used: `VisDrone2019-DET-train.zip`, `VisDrone2019-DET-val.zip`
- Splits: 6,471 train images / 343,205 objects; 548 val images / 38,759
  objects (see `experiments_sci4/dataset_stats_summary.csv`)

### Preparation

1. Extract both archives so that the layout is:

   ```
   dataset/
   ├── VisDrone2019-DET-train/
   │   ├── images/        original frames
   │   └── annotations/   original VisDrone annotation txt files
   └── VisDrone2019-DET-val/
       ├── images/
       └── annotations/
   ```

2. Convert the annotations to YOLO format (writes `labels/` next to
   `annotations/`):

   ```bash
   python src/reconvert_visdrone_unified.py --data-root dataset
   ```

   The converter keeps the 10 detectable categories
   (`pedestrian, people, bicycle, car, van, truck, tricycle,
   awning-tricycle, bus, motor`), maps VisDrone category ids 1–10 to
   YOLO ids 0–9, and skips `ignored regions (0)` / `others (11)`.

3. Edit `dataset/final.yaml`:

   ```yaml
   path: <absolute or relative path to your dataset directory>
   train: VisDrone2019-DET-train/images
   val:   VisDrone2019-DET-val/images
   ```

## UAVDT (cross-dataset generalization)

- Download: <https://sites.google.com/site/daviddo0323/projects/uavdt>
  ("UAVDT" benchmark, DET task; paper: *The Unmanned Aerial Vehicle
  Benchmark: Object Detection and Tracking*, ECCV 2018)
- Classes: 3 vehicle categories — `car (0), truck (1), bus (2)`
- Images are 1024x540 frames extracted from UAV videos

### Preparation

1. Organize the frames into YOLO-style splits:

   ```
   UAVDT/
   ├── train/{images,labels}
   ├── val/{images,labels}
   └── test/{images,labels}
   ```

2. Convert the UAVDT DET annotations to YOLO format (one `class cx cy w h`
   line per object, normalized coordinates; class ids car/truck/bus = 0/1/2)
   and place the resulting `labels/` folders next to `images/`.

3. Edit `dataset/uavdt.yaml` (set `path:` to the UAVDT root):

   ```yaml
   path: <path to UAVDT>
   train: train/images
   val:   val/images
   test:  test/images
   ```

The generalization run in the paper trains YOLOv8m and
YOLOv8m-BiFPN-MixUp on UAVDT with the same 80-epoch recipe used for the main
experiments (see `src/cloud_provenance/`), validating on the UAVDT val split.

## Difficult-subset validation (VisDrone val)

`src/prepare_sci4_experiments.py` computes per-image statistics (object count,
relative areas, occlusion / truncation ratios) and selects the top-100 hardest
val images per attribute. The lists used in the paper are already included in
`experiments_sci4/val_subsets/`:

`crowded_top100.txt`, `small_heavy_top100.txt`, `tiny_heavy_top100.txt`,
`occlusion_top100.txt`, `severe_occlusion_top100.txt`, `truncation_top100.txt`

Each file contains one val image path per line (relative to the dataset
`path:`) and is consumed directly by `src/validate_sci4_subsets.py`, which
generates the per-subset YAML files on the fly.
