import csv,json
from collections import defaultdict
OUT='/home/agipml/samarth.sirsat/DDP/pcb/imaging'
R=[{k:(v if k in('image','cls') else float(v)) for k,v in r.items()} for r in csv.DictReader(open(f'{OUT}/annotations_measured.csv'))]
S=json.load(open('/home/agipml/samarth.sirsat/DDP/pcb/runs/experiments/nq30/summary.json'))
ap=S['per_class_ap50']; mp=S['per_class_map']
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,max(0,int(round(p*(len(v)-1)))))]
per=defaultdict(list)
for r in R: per[r['cls']].append(r)
rows=[]
for c,rs in per.items():
    ms=[r['min_side_px'] for r in rs]; ppm=[r['px_per_mm'] for r in rs]
    rows.append(dict(cls=c,n=len(rs),
        minside_px_med=round(q(ms,.5),1), minside_px_at640=round(q(ms,.5)/2,1),
        minside_mm_med=round(q([r['min_side_mm'] for r in rs],.5),3),
        area_pct_med=round(q([r['area_ratio'] for r in rs],.5)*100,2),
        ppm_spread=round(q(ppm,.9)/q(ppm,.1),1),
        ap50=round(ap[c],3) if c in ap else '', map5095=round(mp[c],3) if c in mp else ''))
rows.sort(key=lambda r:r['minside_px_med'])
ks=list(rows[0].keys())
with open(f'{OUT}/size_vs_performance.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=ks); w.writeheader(); w.writerows(rows)
for r in rows: print(' | '.join(str(r[k]) for k in ks))
# spearman on 8 classes
e=[r for r in rows if r['ap50']!='']
def rank(v):
    s=sorted(range(len(v)),key=lambda i:v[i]); rk=[0]*len(v)
    for i,j in enumerate(s): rk[j]=i+1
    return rk
import math
for metric in ['ap50','map5095']:
    x=rank([r['minside_px_med'] for r in e]); y=rank([r[metric] for r in e])
    n=len(e); d2=sum((a-b)**2 for a,b in zip(x,y)); print(f'spearman(minside_px, {metric}) over {n} classes = %.3f'%(1-6*d2/(n*(n*n-1))))
    x=rank([r['minside_mm_med'] for r in e])
    d2=sum((a-b)**2 for a,b in zip(x,y)); print(f'spearman(minside_mm, {metric}) over {n} classes = %.3f'%(1-6*d2/(n*(n*n-1))))
    x=rank([r['ppm_spread'] for r in e])
    d2=sum((a-b)**2 for a,b in zip(x,y)); print(f'spearman(scale_spread, {metric}) over {n} classes = %.3f'%(1-6*d2/(n*(n*n-1))))
