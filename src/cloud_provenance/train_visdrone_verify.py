
from ultralytics import YOLO
import torch

print("=" * 80)
print("VisDrone 快速验证训练")
print("=" * 80)

print("GPU:", torch.cuda.get_device_name(0))

# 加载模型
model = YOLO("/root/yolov8m.pt")

# 开始训练
results = model.train(

    # 数据集
    data="/root/dataset/final.yaml",

    #=========================
    # 快速验证配置
    #=========================

    epochs=80,

    imgsz=1024,

    batch=8,

    workers=8,

    device=0,


    #=========================
    # 优化器
    #=========================

    optimizer="SGD",

    lr0=0.0005,

    lrf=0.01,

    momentum=0.937,

    weight_decay=0.0005,

    warmup_epochs=5,

    cos_lr=True,


    #=========================
    # 不冻结
    #=========================

    freeze=None,


    #=========================
    # 数据增强
    #=========================

    mosaic=1.0,

    close_mosaic=20,

    mixup=0.03,

    copy_paste=0.0,


    #=========================
    # 几何增强
    #=========================

    degrees=0.0,

    translate=0.1,

    scale=0.5,

    shear=0.0,

    perspective=0.0,


    #=========================
    # 颜色增强
    #=========================

    hsv_h=0.015,

    hsv_s=0.7,

    hsv_v=0.4,


    #=========================
    # 翻转
    #=========================

    fliplr=0.5,

    flipud=0.0,


    #=========================
    # 关键
    #=========================

    rect=False,

    multi_scale=False,

    amp=True,


    #=========================
    # 保存
    #=========================

    patience=30,

    save=True,

    save_period=10,

    plots=True,

    project="/root/runs/detect",

    name="visdrone_verify_80ep",

    verbose=True
)

print("训练结束")