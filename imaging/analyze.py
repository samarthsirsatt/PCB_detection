import os, re, json, csv, math, statistics as st
from collections import Counter, defaultdict
from PIL import Image

ROOT='/home/agipml/samarth.sirsat/DDP/pcb/data/pcb_all_work'
OUT='/home/agipml/samarth.sirsat/DDP/pcb/imaging'
fovpat=re.compile(r'"fov"\s*:\s*([0-9.]+)\s*,\s*"unit"\s*:\s*([0-9]+)')
namepat=re.compile(r'h(\d+(?:\.\d+)?)v(\d+(?:\.\d+)?)mm')

def read_fov(path):
    try:
        im=Image.open(path); ex=im.getexif()
        uc=ex.get_ifd(0x8769).get(37510)
    except Exception:
        return None,None
    if uc is None: return None,None
    b=bytes(uc)
    if b.startswith(b'UNICODE\x00\x00'): s=b[9:].decode('utf-16-le','ignore')
    else: s=b.decode('utf-8','ignore')
    m=fovpat.search(s)
    return (float(m.group(1)), int(m.group(2))) if m else (None,None)

# ---------- EXIF FOV over the whole corpus ----------
corpus=[]
for sub in ['categories','OK PHOTOS_MERGED']:
    for dp,_,fs in os.walk(os.path.join(ROOT,sub)):
        for f in sorted(fs):
            if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
            p=os.path.join(dp,f)
            fov,unit=read_fov(p)
            w,h=Image.open(p).size
            m=namepat.search(f)
            corpus.append(dict(group=sub, file=os.path.relpath(p,ROOT), width=w, height=h,
                               exif_fov_w_mm=fov, exif_unit=unit,
                               name_fov_w_mm=float(m.group(1)) if m else None,
                               name_fov_h_mm=float(m.group(2)) if m else None))
with open(f'{OUT}/image_fov.csv','w',newline='') as fh:
    wtr=csv.DictWriter(fh,fieldnames=list(corpus[0].keys())); wtr.writeheader(); wtr.writerows(corpus)

# cross-check exif vs filename
xc=[(r['exif_fov_w_mm'],r['name_fov_w_mm'],r['name_fov_h_mm']) for r in corpus if r['name_fov_w_mm'] is not None]
agree=sum(1 for a,b,c in xc if a is not None and abs(a-b)<1e-6)
arok=sum(1 for a,b,c in xc if abs(c-b*9/16)<0.002)
print(f'filename-FOV images: {len(xc)}, exif==filename h: {agree}, filename v==h*9/16: {arok}')
print('exif fov present:', sum(1 for r in corpus if r['exif_fov_w_mm'] is not None),'/',len(corpus))
print('exif units:', Counter(r['exif_unit'] for r in corpus))

# ---------- COCO ----------
coco=json.load(open(f'{ROOT}/coco/annotations/instances_default.json'))
cats={c['id']:c['name'] for c in coco['categories']}
imgs={i['id']:i for i in coco['images']}
fovmap={r['file']:r['exif_fov_w_mm'] for r in corpus}

recs=[]
for a in coco['annotations']:
    im=imgs[a['image_id']]; W,H=im['width'],im['height']
    x,y,bw,bh=a['bbox']
    fn=im['file_name']
    fov=fovmap.get(fn)
    ppm = W/fov if fov else None
    r=dict(ann_id=a['id'], image=fn, cls=cats[a['category_id']], img_w=W, img_h=H,
           bbox_x=x, bbox_y=y, bbox_w_px=bw, bbox_h_px=bh, bbox_area_px2=bw*bh,
           w_ratio=bw/W, h_ratio=bh/H, area_ratio=(bw*bh)/(W*H),
           fov_w_mm=fov, fov_h_mm=(fov*H/W) if fov else None,
           px_per_mm=ppm,
           bbox_w_mm=(bw/ppm) if ppm else None, bbox_h_mm=(bh/ppm) if ppm else None,
           bbox_area_mm2=(bw*bh/ppm**2) if ppm else None,
           diag_px=math.hypot(bw,bh), min_side_px=min(bw,bh),
           min_side_mm=(min(bw,bh)/ppm) if ppm else None)
    recs.append(r)
with open(f'{OUT}/annotations_measured.csv','w',newline='') as fh:
    wtr=csv.DictWriter(fh,fieldnames=list(recs[0].keys())); wtr.writeheader(); wtr.writerows(recs)

def stats(v):
    v=sorted(v)
    n=len(v)
    q=lambda p: v[min(n-1,max(0,int(round(p*(n-1)))))]
    return dict(n=n,min=v[0],p25=q(.25),median=q(.5),mean=sum(v)/n,p75=q(.75),max=v[-1],
                sd=st.pstdev(v) if n>1 else 0.0)

per=defaultdict(list)
for r in recs: per[r['cls']].append(r)
summary={}
for c,rs in sorted(per.items()):
    d={'count':len(rs),'n_images':len({r['image'] for r in rs}),
       'fov_min_mm':min(r['fov_w_mm'] for r in rs),'fov_max_mm':max(r['fov_w_mm'] for r in rs),
       'fov_median_mm':stats([r['fov_w_mm'] for r in rs])['median'],
       'ppm_median':stats([r['px_per_mm'] for r in rs])['median']}
    for f in ['bbox_w_px','bbox_h_px','bbox_area_px2','area_ratio','bbox_w_mm','bbox_h_mm','bbox_area_mm2','min_side_px','min_side_mm']:
        d[f]=stats([r[f] for r in rs])
    summary[c]=d
json.dump(summary,open(f'{OUT}/per_class_stats.json','w'),indent=2)

with open(f'{OUT}/per_class_stats.csv','w',newline='') as fh:
    w=csv.writer(fh)
    w.writerow(['class','count','fov_med_mm','ppm_med','w_px_med','h_px_med','minside_px_med','minside_px_min',
                'w_mm_med','h_mm_med','minside_mm_med','minside_mm_min','area_px2_med','area_mm2_med','area_ratio_med_pct'])
    for c,d in sorted(summary.items(),key=lambda kv:kv[1]['min_side_px']['median']):
        w.writerow([c,d['count'],round(d['fov_median_mm'],2),round(d['ppm_median'],2),
            round(d['bbox_w_px']['median'],1),round(d['bbox_h_px']['median'],1),
            round(d['min_side_px']['median'],1),round(d['min_side_px']['min'],1),
            round(d['bbox_w_mm']['median'],3),round(d['bbox_h_mm']['median'],3),
            round(d['min_side_mm']['median'],3),round(d['min_side_mm']['min'],4),
            round(d['bbox_area_px2']['median'],0),round(d['bbox_area_mm2']['median'],4),
            round(d['area_ratio']['median']*100,3)])

allfov=[r['fov_w_mm'] for r in recs]
print('\nANNOTATED IMAGES FOV: n=%d min=%.2f med=%.2f max=%.2f'%(len(allfov),min(allfov),stats(allfov)['median'],max(allfov)))
print('annotated-image FOV histogram:',sorted(Counter(allfov).items()))
okf=[r['exif_fov_w_mm'] for r in corpus if r['group']=='OK PHOTOS_MERGED']
print('OK FOV: min=%.2f med=%.2f max=%.2f'%(min(okf),stats(okf)['median'],max(okf)))
print('\nper class (sorted by median min-side px):')
print(open(f'{OUT}/per_class_stats.csv').read())
