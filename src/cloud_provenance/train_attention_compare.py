from pathlib import Path
from ultralytics import YOLO

root = Path('/root/autodl-tmp/ifball_uav')
experiments = [
    ('YOLOv8m_CBAM', root / 'src/attention_compare/yolov8m-cbam.yaml', 'visdrone_yolov8m_cbam_80ep'),
    ('YOLOv8m_ECA', root / 'src/attention_compare/yolov8m-eca.yaml', 'visdrone_yolov8m_eca_80ep'),
    ('YOLOv8m_C3TR_Transformer', root / 'src/attention_compare/yolov8m-c3tr.yaml', 'visdrone_yolov8m_c3tr_80ep'),
]
for label, model_yaml, name in experiments:
    run_dir = root / 'runs/attention_compare' / name
    done = run_dir / 'weights/best.pt'
    if done.exists() and (run_dir / 'results.csv').exists():
        print(f'=== SKIP {label}: existing run found at {run_dir} ===', flush=True)
        continue
    print(f'=== START {label} ===', flush=True)
    print(f'model={model_yaml}', flush=True)
    YOLO(str(model_yaml)).train(
        data=str(root / 'dataset/final.yaml'),
        epochs=80,
        imgsz=1024,
        batch=8,
        device=0,
        workers=8,
        project=str(root / 'runs/attention_compare'),
        name=name,
        pretrained=str(root / 'src/yolov8m.pt'),
        optimizer='AdamW',
        lr0=0.0005,
        warmup_epochs=0,
        close_mosaic=0,
        mixup=0.0,
        cos_lr=False,
        save_period=10,
        patience=60,
        exist_ok=True,
        amp=False,
    )
    print(f'=== FINISH {label} ===', flush=True)
print('=== ALL ATTENTION COMPARISON EXPERIMENTS FINISHED ===', flush=True)
