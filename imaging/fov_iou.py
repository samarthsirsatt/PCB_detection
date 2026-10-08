import os,csv,json
from pathlib import Path
BASE=Path('/home/agipml/samarth.sirsat/DDP/pcb/scratch/exp_data/neg120')
OUT=Path('/home/agipml/samarth.sirsat/DDP/pcb/imaging')
W='/home/agipml/samarth.sirsat/DDP/pcb/runs/experiments/nq30/weights/best.pt'
NAMES=["Component Liftup","Component Missing","Component No Solder","Component Solder Dry",
       "Polarity Wrong","RYB Wrong Sequence","Solder Ball","Solder Short"]
rows=[r for r in csv.DictReader(open(OUT/'val_fov_bins.csv')) if r['is_negative']=='False']
from ultralytics import RTDETR
m=RTDETR(W)
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
res=[]
for k,r in enumerate(rows):
    f=r['file']; st=Path(f).stem
    p=str(BASE/'images/val'/f)
    pr=m.predict(p,imgsz=640,conf=0.10,device='cpu',verbose=False)[0]
    W_,H_=1280,720
    preds=[(int(c),float(cf),list(map(float,b))) for c,cf,b in
           zip(pr.boxes.cls.tolist(),pr.boxes.conf.tolist(),pr.boxes.xyxy.tolist())]
    for line in open(BASE/'labels/val'/(st+'.txt')).read().split('\n'):
        if not line.strip(): continue
        c,cx,cy,w,h=line.split(); c=int(c); cx,cy,w,h=[float(x) for x in (cx,cy,w,h)]
        gt=[(cx-w/2)*W_,(cy-h/2)*H_,(cx+w/2)*W_,(cy+h/2)*H_]
        same=[(cf,iou(gt,b)) for cc,cf,b in preds if cc==c]
        any_=[(cf,iou(gt,b)) for cc,cf,b in preds]
        best=max((i for _,i in same),default=0.0)
        besta=max((i for _,i in any_),default=0.0)
        res.append(dict(file=f,cls=NAMES[c],fov_mm=float(r['fov_mm']),px_per_mm=float(r['px_per_mm']),
                        bin=r['bin'],gt_w_px=w*W_,gt_h_px=h*H_,gt_minside_px=min(w*W_,h*H_),
                        gt_minside_mm=min(w*W_,h*H_)/float(r['px_per_mm']),
                        best_iou_sameclass=round(best,4),best_iou_anyclass=round(besta,4),
                        detected_050=int(best>=0.5),detected_075=int(best>=0.75)))
    if k%20==0: print(k,flush=True)
with open(OUT/'val_gt_iou.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
print('gt boxes',len(res))
