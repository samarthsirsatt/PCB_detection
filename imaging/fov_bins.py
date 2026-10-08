import os,re,csv,json
from collections import Counter,defaultdict
from PIL import Image
VAL='/home/agipml/samarth.sirsat/DDP/pcb/scratch/exp_data/neg120/images/val'
LAB='/home/agipml/samarth.sirsat/DDP/pcb/scratch/exp_data/neg120/labels/val'
NAMES=["Component Liftup","Component Missing","Component No Solder","Component Solder Dry",
       "Polarity Wrong","RYB Wrong Sequence","Solder Ball","Solder Short"]
pat=re.compile(r'"fov"\s*:\s*([0-9.]+)')
def fov(p):
    ex=Image.open(p).getexif().get_ifd(0x8769).get(37510)
    b=bytes(ex); s=b[9:].decode('utf-16-le','ignore') if b.startswith(b'UNICODE\x00\x00') else b.decode('utf-8','ignore')
    m=pat.search(s); return float(m.group(1)) if m else None
BINS=[(1,10,'1-10 mm'),(10,15,'10-15 mm'),(15,20,'15-20 mm'),(20,25,'20-25 mm'),(25,30,'25-30 mm'),(30,40.001,'30-40 mm')]
def which(f):
    for lo,hi,n in BINS:
        if lo<=f<hi: return n
    return 'other'
rows=[]
for f in sorted(os.listdir(VAL)):
    p=os.path.join(VAL,f); v=fov(p)
    lab=os.path.join(LAB,os.path.splitext(f)[0]+'.txt')
    ls=[l for l in open(lab).read().split('\n') if l.strip()]
    cls=[NAMES[int(l.split()[0])] for l in ls]
    rows.append(dict(file=f,fov_mm=v,px_per_mm=1280/v,bin=which(v),n_boxes=len(ls),classes=';'.join(cls),
                     is_negative=len(ls)==0))
with open('/home/agipml/samarth.sirsat/DDP/pcb/imaging/val_fov_bins.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('val images',len(rows),'boxes',sum(r['n_boxes'] for r in rows),'negatives',sum(r['is_negative'] for r in rows))
print('\nbin | imgs | pos_imgs | negs | boxes | px/mm range | classes present')
order=[n for _,_,n in BINS]+['other']
for b in order:
    rs=[r for r in rows if r['bin']==b]
    if not rs: print(b,'EMPTY'); continue
    cc=Counter(c for r in rs for c in r['classes'].split(';') if c)
    fv=[r['fov_mm'] for r in rs]
    print('%-9s %4d %4d %4d %5d  %.1f-%.1f  %s'%(b,len(rs),sum(1 for r in rs if not r['is_negative']),
        sum(1 for r in rs if r['is_negative']),sum(r['n_boxes'] for r in rs),
        1280/max(fv),1280/min(fv), dict(cc)))
print('\nFOV values present:',sorted(Counter(r['fov_mm'] for r in rows).items()))
