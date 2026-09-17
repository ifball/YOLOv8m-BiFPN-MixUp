from ultralytics import YOLO
import torch

print("=" * 80)
print("VisDrone YOLOv8m-BiFPN 消融实验")
print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("/root/src/yolov8m-bifpn.yaml")
model.load("/root/yolov8m.pt")

results = model.train(
    data="/root/dataset/final.yaml",

    epochs=80,
    imgsz=1024,
    batch=8,
    workers=8,
    device=0,

    optimizer="SGD",
    lr0=0.0005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,

    freeze=None,

    mosaic=1.0,
    close_mosaic=20,
    mixup=0.0,
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
    amp=True,

    patience=30,
    save=True,
    save_period=10,
    plots=True,

    project="/root/runs/detect",
    name="visdrone_yolov8m_bifpn_80ep",

    verbose=True
)

print("YOLOv8m-BiFPN训练完成")
