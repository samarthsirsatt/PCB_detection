import os,csv,json,shutil,sys
from pathlib import Path
BASE=Path('/home/agipml/samarth.sirsat/DDP/pcb/scratch/exp_data/neg120')
OUT=Path('/home/agipml/samarth.sirsat/DDP/pcb/imaging')
WORK=Path('/home/agipml/samarth.sirsat/DDP/pcb/scratch/fov_eval')
W='/home/agipml/samarth.sirsat/DDP/pcb/runs/experiments/nq30/weights/best.pt'
NAMES=["Component Liftup","Component Missing","Component No Solder","Component Solder Dry",
       "Polarity Wrong","RYB Wrong Sequence","Solder Ball","Solder Short"]
rows=list(csv.DictReader(open(OUT/'val_fov_bins.csv')))
groups={}
for r in rows: groups.setdefault(r['bin'],[]).append(r)
groups['ALL']=rows

import yaml
from ultralytics import RTDETR
model=RTDETR(W)
res_all={}
for g,rs in list(groups.items()):
    tag=g.replace(' ','').replace('-','_')
    d=WORK/tag; shutil.rmtree(d,ignore_errors=True)
    (d/'images/val').mkdir(parents=True); (d/'labels/val').mkdir(parents=True)
    for r in rs:
        f=r['file']; st=Path(f).stem
        os.symlink(BASE/'images/val'/f, d/'images/val'/f)
        shutil.copy(BASE/'labels/val'/(st+'.txt'), d/'labels/val'/(st+'.txt'))
    (d/'data.yaml').write_text(yaml.safe_dump({'path':str(d),'train':'images/val','val':'images/val',
        'names':{i:n for i,n in enumerate(NAMES)}},sort_keys=False))
    print('=== evaluating',g,len(rs),'images',flush=True)
    r_=model.val(data=str(d/'data.yaml'),split='val',imgsz=640,device='cpu',
                 project=str(WORK/'runs'),name=tag,exist_ok=True,plots=False,verbose=False)
    b=r_.box
    ap50={}; map_={}
    try:
        idx=list(b.ap_class_index); a50=list(b.ap50)
        ap50={NAMES[int(c)]:float(a50[j]) for j,c in enumerate(idx) if j<len(a50)}
        maps=list(b.maps)
        map_={NAMES[int(c)]:float(maps[int(c)]) for c in idx}
    except Exception as e: print('warn',e)
    res_all[g]=dict(n_images=len(rs), n_pos_images=sum(1 for r in rs if r['is_negative']=='False'),
        n_neg_images=sum(1 for r in rs if r['is_negative']=='True'),
        n_boxes=sum(int(r['n_boxes']) for r in rs),
        map50=float(b.map50), map=float(b.map), precision=float(b.mp), recall=float(b.mr),
        per_class_ap50=ap50, per_class_map=map_)
    print(g, json.dumps({k:round(res_all[g][k],4) for k in ('map50','map','precision','recall')}),flush=True)
json.dump(res_all,open(OUT/'fov_bin_results.json','w'),indent=2)
print('done')
