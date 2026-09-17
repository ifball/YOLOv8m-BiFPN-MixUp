from ultralytics import YOLO
import torch

print("=" * 80)
print("YOLOv8l-BiFPN 80ep")
print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("/root/src/yolov8l-bifpn.yaml")
model.load("/root/yolov8l.pt")

model.train(
    data="/root/dataset/final.yaml",
    epochs=80,
    imgsz=1024,
    batch=4,
    workers=8,
    device=0,

    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    warmup_epochs=5,
    cos_lr=True,
    patience=40,

    mosaic=1.0,
    close_mosaic=15,
    mixup=0.1,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    amp=True,
    plots=True,
    save=True,
    save_period=10,

    project="/root/runs/detect",
    name="compare_yolov8l_bifpn_80ep",
    verbose=True
)

print("YOLOv8l-BiFPN 训练完成")
