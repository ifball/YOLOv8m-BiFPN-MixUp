from pathlib import Path
from ultralytics import YOLO, RTDETR

root = Path('/root/autodl-tmp/ifball_uav')
data = str(root / 'dataset/final.yaml')
project = str(root / 'runs/frontier_compare')

common = dict(
    data=data,
    epochs=80,
    imgsz=1024,
    workers=8,
    device=0,
    amp=False,
    project=project,
    exist_ok=True,
    plots=True,
    save=True,
    save_period=-1,
    verbose=True,
)

print('=== VisDrone frontier compare: YOLO11m ===', flush=True)
YOLO('yolo11m.pt').train(
    **common,
    batch=8,
    name='visdrone_yolo11m_80ep',
)

print('=== VisDrone frontier compare: RT-DETR-l ===', flush=True)
RTDETR('rtdetr-l.pt').train(
    **common,
    batch=4,
    name='visdrone_rtdetr_l_80ep',
)

print('=== VisDrone frontier compare finished ===', flush=True)
