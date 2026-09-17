from pathlib import Path
from ultralytics import RTDETR

root = Path('/root/autodl-tmp/ifball_uav')
data = str(root / 'dataset/final.yaml')
project = str(root / 'runs/frontier_compare')

print('=== VisDrone frontier compare: RT-DETR-l only ===', flush=True)
RTDETR(str(root / 'rtdetr-l.pt')).train(
    data=data,
    epochs=80,
    imgsz=1024,
    batch=2,
    workers=8,
    device=0,
    amp=False,
    project=project,
    name='visdrone_rtdetr_l_80ep',
    exist_ok=True,
    plots=True,
    save=True,
    save_period=-1,
    verbose=True,
)
print('=== RT-DETR-l finished ===', flush=True)
