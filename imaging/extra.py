import csv, json, math, statistics as st
from collections import Counter, defaultdict
OUT='/home/agipml/samarth.sirsat/DDP/pcb/imaging'
R=[{k:(float(v) if v not in('','None') and k not in('image','cls') else v) for k,v in r.items()} for r in csv.DictReader(open(f'{OUT}/annotations_measured.csv'))]

ppm=[r['px_per_mm'] for r in R]
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,max(0,int(round(p*(len(v)-1)))))]
print('px/mm over annotated: min %.1f p25 %.1f med %.1f p75 %.1f max %.1f'%(min(ppm),q(ppm,.25),q(ppm,.5),q(ppm,.75),max(ppm)))
print('unique px/mm values:',len(set(round(x,3) for x in ppm)))

# COCO size buckets (px)
b=Counter()
for r in R:
    a=r['bbox_area_px2']
    b['small(<32^2)' if a<1024 else 'medium(<96^2)' if a<9216 else 'large']+=1
print('COCO px buckets:',dict(b), 'of',len(R))
bc=defaultdict(Counter)
for r in R:
    a=r['bbox_area_px2']
    bc[r['cls']]['small' if a<1024 else 'medium' if a<9216 else 'large']+=1
for c in sorted(bc): print('  ',c,dict(bc[c]))

# correlation px area vs mm area (log)
import math
x=[math.log(r['bbox_area_px2']) for r in R]; y=[math.log(r['bbox_area_mm2']) for r in R]
mx,my=sum(x)/len(x),sum(y)/len(y)
num=sum((a-mx)*(bb-my) for a,bb in zip(x,y)); den=math.sqrt(sum((a-mx)**2 for a in x)*sum((bb-my)**2 for bb in y))
print('log-log corr(px area, mm area) = %.3f'%(num/den))
xf=[math.log(r['fov_w_mm']) for r in R]
mf=sum(xf)/len(xf)
num2=sum((a-mf)*(bb-my) for a,bb in zip(xf,y)); den2=math.sqrt(sum((a-mf)**2 for a in xf)*sum((bb-my)**2 for bb in y))
print('log-log corr(FOV, mm area)     = %.3f'%(num2/den2))
num3=sum((a-mf)*(bb-mx) for a,bb in zip(xf,x)); den3=math.sqrt(sum((a-mf)**2 for a in xf)*sum((bb-mx)**2 for bb in x))
print('log-log corr(FOV, px area)     = %.3f'%(num3/den3))

# smallest annotations
print('\n10 smallest by min_side_px:')
for r in sorted(R,key=lambda r:r['min_side_px'])[:10]:
    print('  %-22s %5.1fpx  fov %5.2fmm  %.1f px/mm  -> %.3f mm  %s'%(r['cls'],r['min_side_px'],r['fov_w_mm'],r['px_per_mm'],r['min_side_mm'],r['image'].split('/')[-1][:38]))
print('\n10 smallest by min_side_mm:')
for r in sorted(R,key=lambda r:r['min_side_mm'])[:10]:
    print('  %-22s %.4f mm  (%5.1f px, fov %5.2f mm, %.1f px/mm) %s'%(r['cls'],r['min_side_mm'],r['min_side_px'],r['fov_w_mm'],r['px_per_mm'],r['image'].split('/')[-1][:38]))

# Solder Ball detail
sb=[r for r in R if r['cls']=='Solder Ball']
for f in ['min_side_px','min_side_mm','bbox_area_mm2','fov_w_mm','px_per_mm']:
    v=[r[f] for r in sb]
    print('SolderBall %-14s min %.4f p25 %.4f med %.4f p75 %.4f max %.4f'%(f,min(v),q(v,.25),q(v,.5),q(v,.75),max(v)))

# per class p10 of min_side_mm  (robust small end)
print('\nclass, p10 min_side_mm, med min_side_mm')
per=defaultdict(list)
for r in R: per[r['cls']].append(r['min_side_mm'])
for c,v in sorted(per.items(), key=lambda kv: q(kv[1],.5)):
    print('  %-22s p10=%.3f  med=%.3f  n=%d'%(c,q(v,.10),q(v,.5),len(v)))
