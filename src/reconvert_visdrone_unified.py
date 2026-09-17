
import argparse
import os

parser = argparse.ArgumentParser(description="Convert VisDrone2019-DET annotations to YOLO format.")
parser.add_argument("--data-root", type=str, default="dataset",
                    help="Directory that contains VisDrone2019-DET-train / VisDrone2019-DET-val.")
args = parser.parse_args()

splits = ["train", "val"]

mapping = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

for split in splits:
    root = os.path.join(args.data_root, f"VisDrone2019-DET-{split}")
    ann_dir = os.path.join(root, "annotations")
    img_dir = os.path.join(root, "images")
    label_dir = os.path.join(root, "labels")

    os.makedirs(label_dir, exist_ok=True)

    for old in os.listdir(label_dir):
        if old.endswith(".txt"):
            os.remove(os.path.join(label_dir, old))

    count = 0
    skipped = 0

    for ann_name in os.listdir(ann_dir):
        if not ann_name.endswith(".txt"):
            continue

        img_name = ann_name.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        import cv2
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        out_lines = []
        ann_path = os.path.join(ann_dir, ann_name)

        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")

                if len(parts) < 6:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                bw = float(parts[2])
                bh = float(parts[3])
                category = int(parts[5])

                if category not in mapping:
                    skipped += 1
                    continue

                cls = mapping[category]

                if bw <= 0 or bh <= 0:
                    continue

                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                    continue

                out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        save_path = os.path.join(label_dir, ann_name)

        with open(save_path, "w") as f:
            f.writelines(out_lines)

        count += len(out_lines)

    print(split, "converted labels:", count, "skipped:", skipped)

print("统一转换完成")
