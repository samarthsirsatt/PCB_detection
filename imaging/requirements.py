import csv,json,math
from collections import defaultdict
OUT='/home/agipml/samarth.sirsat/DDP/pcb/imaging'
R=[{k:(v if k in('image','cls') else float(v)) for k,v in r.items()} for r in csv.DictReader(open(f'{OUT}/annotations_measured.csv'))]
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,max(0,int(round(p*(len(v)-1)))))]
per=defaultdict(list)
for r in R: per[r['cls']].append(r['min_side_mm'])

targets=[3,5,10,15,20]
rows=[]
print('required px/mm to place N pixels across the SHORT side of a defect')
print('class | p10 mm | med mm | ' + ' | '.join(f'{t}px' for t in targets))
for c,v in sorted(per.items(), key=lambda kv:q(kv[1],.10)):
    p10=q(v,.10); med=q(v,.5)
    rows.append([c,round(p10,3),round(med,3)]+[round(t/p10,1) for t in targets])
    print('%-22s %6.3f %6.3f  '%(c,p10,med)+'  '.join('%8.1f'%(t/p10) for t in targets))
with open(f'{OUT}/required_px_per_mm.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['class','p10_min_side_mm','median_min_side_mm']+[f'req_px_per_mm_for_{t}px' for t in targets]); w.writerows(rows)

print('\nFOV (mm) implied by sensor width at given px/mm  [FOV = width_px / px_per_mm]')
widths=[1280,1920,2448,4096,5472]
ppms=[32,51.2,64,71.1,100,150,200,300]
print('px/mm |'+''.join('%9d'%w for w in widths))
tbl=[]
for p in ppms:
    print('%6.1f|'%p+''.join('%9.1f'%(w/p) for w in widths)); tbl.append([p]+[round(w/p,2) for w in widths])
with open(f'{OUT}/fov_vs_resolution.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['px_per_mm']+[f'fov_mm_at_{x}px' for x in widths]); w.writerows(tbl)

print('\nsensor px needed to cover a board of side S at px/mm P  (S x P)')
for S in [50,100,150,200]:
    print('  %3d mm: '%S+'  '.join('%.1f MP @ %g px/mm'%((S*p)*(S*p)/1e6,p) for p in [32,71.1,100,200]))
