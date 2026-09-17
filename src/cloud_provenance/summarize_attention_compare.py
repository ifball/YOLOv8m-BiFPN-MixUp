import csv
from pathlib import Path
root=Path('/root/autodl-tmp/ifball_uav')
runs={
 'YOLOv8m+CBAM': root/'runs/attention_compare/visdrone_yolov8m_cbam_80ep/results.csv',
 'YOLOv8m+ECA': root/'runs/attention_compare/visdrone_yolov8m_eca_80ep/results.csv',
 'YOLOv8m+C3TR': root/'runs/attention_compare/visdrone_yolov8m_c3tr_80ep/results.csv',
}
out=root/'experiments_sci4/attention_compare/attention_compare_summary.csv'
rows=[]
for name,p in runs.items():
    if not p.exists():
        rows.append({'method':name,'status':'missing'})
        continue
    data=list(csv.DictReader(p.open(newline='')))
    def f(r,k):
        try: return float(r.get(k,''))
        except: return -1
    best50=max(data,key=lambda r:f(r,'metrics/mAP50(B)'))
    best95=max(data,key=lambda r:f(r,'metrics/mAP50-95(B)'))
    last=data[-1]
    rows.append({
        'method':name,'status':'done','epochs':len(data),
        'best_mAP50_epoch':best50['epoch'],'P_at_best50':best50['metrics/precision(B)'],'R_at_best50':best50['metrics/recall(B)'],'mAP50':best50['metrics/mAP50(B)'],'mAP50-95_at_best50':best50['metrics/mAP50-95(B)'],
        'best_mAP50-95_epoch':best95['epoch'],'P_at_best95':best95['metrics/precision(B)'],'R_at_best95':best95['metrics/recall(B)'],'mAP50_at_best95':best95['metrics/mAP50(B)'],'mAP50-95':best95['metrics/mAP50-95(B)'],
        'last_epoch':last['epoch'],'last_P':last['metrics/precision(B)'],'last_R':last['metrics/recall(B)'],'last_mAP50':last['metrics/mAP50(B)'],'last_mAP50-95':last['metrics/mAP50-95(B)'],
    })
keys=sorted({k for r in rows for k in r})
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
print(out)
for r in rows: print(r)
