import csv, json
from pathlib import Path
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops
root=Path('/root/autodl-tmp/ifball_uav')
base=root/'experiments_sci4/efficiency_compare/efficiency_comparison_val1024.csv'
models={
 'YOLOv8m baseline': root/'runs/detect/visdrone_verify_80ep2/weights/best.pt',
 'YOLOv8m+CBAM': root/'runs/attention_compare/visdrone_yolov8m_cbam_80ep/weights/best.pt',
 'YOLOv8m+ECA': root/'runs/attention_compare/visdrone_yolov8m_eca_80ep/weights/best.pt',
 'YOLOv8m+C3TR': root/'runs/attention_compare/visdrone_yolov8m_c3tr_80ep/weights/best.pt',
 'YOLOv8m+BiFPN+MixUp': root/'runs/detect_sci4/sci4_B4_bifpn_mixup/weights/best.pt',
 'YOLO11m': root/'runs/frontier_compare/visdrone_yolo11m_80ep/weights/best.pt',
 'RT-DETR-l': root/'runs/frontier_compare/visdrone_rtdetr_l_80ep/weights/best.pt',
}
info={}
for name,w in models.items():
    cls=RTDETR if 'rtdetr' in str(w).lower() else YOLO
    m=cls(str(w)).model
    params=sum(p.numel() for p in m.parameters())
    layers=len(list(m.modules()))
    try: flops=get_flops(m, imgsz=1024)
    except Exception: flops=0.0
    info[name]={'params_M':params/1e6, 'GFLOPs':flops, 'module_layers':layers}
rows=list(csv.DictReader(base.open(newline='')))
for r in rows:
    inf=info.get(r['method'],{})
    r['params_M']=f"{inf.get('params_M',''):.3f}" if inf else r.get('params_M','')
    r['GFLOPs']=f"{inf.get('GFLOPs',''):.1f}" if inf and inf.get('GFLOPs') else r.get('GFLOPs','')
    r['FPS']=f"{float(r['FPS']):.2f}" if r.get('FPS') else ''
    for k in ['precision','recall','mAP50','mAP50-95','preprocess_ms','inference_ms','postprocess_ms','total_ms_per_image']:
        if r.get(k): r[k]=f"{float(r[k]):.4f}" if 'mAP' in k or k in ['precision','recall'] else f"{float(r[k]):.2f}"
# Add deltas vs final
final=next(r for r in rows if r['method']=='YOLOv8m+BiFPN+MixUp')
f50=float(final['mAP50']); f95=float(final['mAP50-95'])
for r in rows:
    r['delta_mAP50_vs_final']=f"{float(r['mAP50'])-f50:+.4f}" if r.get('mAP50') else ''
    r['delta_mAP50-95_vs_final']=f"{float(r['mAP50-95'])-f95:+.4f}" if r.get('mAP50-95') else ''
out=root/'experiments_sci4/efficiency_compare/final_efficiency_comparison_val1024.csv'
fields=['method','params_M','GFLOPs','precision','recall','mAP50','mAP50-95','delta_mAP50_vs_final','delta_mAP50-95_vs_final','inference_ms','total_ms_per_image','FPS','preprocess_ms','postprocess_ms','weight']
with out.open('w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows([{k:r.get(k,'') for k in fields} for r in rows])
print(out)
print(json.dumps([{k:r.get(k,'') for k in fields[:-1]} for r in rows], ensure_ascii=False, indent=2))
