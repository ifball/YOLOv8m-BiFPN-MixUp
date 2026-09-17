import csv, io, re, contextlib, json
from pathlib import Path
from ultralytics import YOLO, RTDETR

root=Path('/root/autodl-tmp/ifball_uav')
data=str(root/'dataset/final.yaml')
outdir=root/'experiments_sci4/efficiency_compare'
outdir.mkdir(parents=True, exist_ok=True)

candidates={
 'YOLOv8m baseline': [root/'runs/detect/visdrone_yolov8m_80ep/weights/best.pt', root/'runs/detect/visdrone_yolov8m_baseline_80ep/weights/best.pt', root/'runs/detect_sci4/visdrone_yolov8m_baseline_80ep/weights/best.pt'],
 'YOLOv8m+CBAM': [root/'runs/attention_compare/visdrone_yolov8m_cbam_80ep/weights/best.pt'],
 'YOLOv8m+ECA': [root/'runs/attention_compare/visdrone_yolov8m_eca_80ep/weights/best.pt'],
 'YOLOv8m+C3TR': [root/'runs/attention_compare/visdrone_yolov8m_c3tr_80ep/weights/best.pt'],
 'YOLOv8m+BiFPN+MixUp': [root/'runs/detect_sci4/sci4_B4_bifpn_mixup/weights/best.pt'],
 'YOLO11m': [root/'runs/frontier_compare/visdrone_yolo11m_80ep/weights/best.pt'],
 'RT-DETR-l': [root/'runs/frontier_compare/visdrone_rtdetr_l_80ep/weights/best.pt'],
}
# add baseline fallback by search
if not any(p.exists() for p in candidates['YOLOv8m baseline']):
    for p in root.rglob('*/weights/best.pt'):
        s=str(p).lower()
        if 'yolov8m' in s and 'baseline' in s:
            candidates['YOLOv8m baseline'].append(p)

def choose(paths):
    for p in paths:
        if p.exists(): return p
    return None

def get_info(model):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            model.model.info(verbose=False, imgsz=1024)
        except TypeError:
            model.model.info(verbose=False)
    text=buf.getvalue()
    # examples: "YOLOv8m summary: 169 layers, 25,902,640 parameters, ... 79.3 GFLOPs"
    params=gflops=layers=None
    m=re.search(r'(\d[\d,]*) parameters', text)
    if m: params=int(m.group(1).replace(',',''))
    m=re.search(r'(\d+(?:\.\d+)?) GFLOPs', text)
    if m: gflops=float(m.group(1))
    m=re.search(r'(\d+) layers', text)
    if m: layers=int(m.group(1))
    return layers, params, gflops, text.strip()

# metric best values from earlier csv summaries if available
known={}
sum_path=root/'experiments_sci4/all_ablation_summary_best.csv'
if sum_path.exists():
    for r in csv.DictReader(sum_path.open(newline='')):
        known[r['name']]=r

rows=[]
for name, paths in candidates.items():
    weight=choose(paths)
    if not weight:
        rows.append({'method':name,'status':'missing_weight'})
        continue
    cls=RTDETR if 'rtdetr' in str(weight).lower() else YOLO
    model=cls(str(weight))
    layers, params, gflops, info_text=get_info(model)
    print('VALIDATING', name, weight, flush=True)
    metrics=model.val(data=data, imgsz=1024, batch=8, device=0, workers=8, project=str(outdir/'val_runs'), name=name.replace('+','_').replace(' ','_').replace('-','_'), exist_ok=True, verbose=False)
    speed=getattr(metrics, 'speed', {}) or {}
    preprocess=float(speed.get('preprocess', 0) or 0)
    inference=float(speed.get('inference', 0) or 0)
    postprocess=float(speed.get('postprocess', 0) or 0)
    total=preprocess+inference+postprocess
    fps=1000/total if total>0 else ''
    box=getattr(metrics, 'box', None)
    p=r=map50=map95=''
    if box is not None:
        p=float(getattr(box,'mp',0)); r=float(getattr(box,'mr',0)); map50=float(getattr(box,'map50',0)); map95=float(getattr(box,'map',0))
    rows.append({
        'method':name, 'status':'done', 'weight':str(weight), 'layers':layers, 'params_M': params/1e6 if params else '', 'GFLOPs':gflops,
        'precision':p, 'recall':r, 'mAP50_val':map50, 'mAP50-95_val':map95,
        'preprocess_ms':preprocess, 'inference_ms':inference, 'postprocess_ms':postprocess, 'total_ms_per_image':total, 'FPS':fps,
        'info_text':info_text,
    })

csv_path=outdir/'efficiency_comparison_val1024.csv'
fields=['method','status','layers','params_M','GFLOPs','precision','recall','mAP50_val','mAP50-95_val','preprocess_ms','inference_ms','postprocess_ms','total_ms_per_image','FPS','weight']
with csv_path.open('w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k:r.get(k,'') for k in fields})
print('WROTE', csv_path)
print(json.dumps(rows, ensure_ascii=False, indent=2))
