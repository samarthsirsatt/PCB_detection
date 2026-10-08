import os, re, json, csv, collections
from PIL import Image
from PIL.ExifTags import TAGS

ROOT='/home/agipml/samarth.sirsat/DDP/pcb/data/pcb_all_work'
OUT='/home/agipml/samarth.sirsat/DDP/pcb/imaging'
pat=re.compile(r'h(\d+(?:\.\d+)?)v(\d+(?:\.\d+)?)mm')

rows=[]
for sub in ['categories','OK PHOTOS_MERGED']:
    base=os.path.join(ROOT,sub)
    for dirpath,_,files in os.walk(base):
        for f in files:
            if not f.lower().endswith(('.jpg','.jpeg','.png','.bmp')): continue
            p=os.path.join(dirpath,f)
            try:
                im=Image.open(p); w,h=im.size; fmt=im.format
                ex=im.getexif()
                exif={TAGS.get(k,k):v for k,v in ex.items()}
                try:
                    for k,v in ex.get_ifd(0x8769).items():
                        exif[TAGS.get(k,k)]=v
                except Exception: pass
            except Exception as e:
                rows.append(dict(group=sub,file=os.path.relpath(p,ROOT),error=str(e))); continue
            m=pat.search(f)
            rows.append(dict(group=sub, file=os.path.relpath(p,ROOT), width=w, height=h,
                format=fmt, bytes=os.path.getsize(p),
                fov_w_mm=float(m.group(1)) if m else '', fov_h_mm=float(m.group(2)) if m else '',
                exif_keys=';'.join(sorted(str(k) for k in exif.keys())),
                make=exif.get('Make',''), model=exif.get('Model',''),
                focal=exif.get('FocalLength',''), fnum=exif.get('FNumber',''),
                exp=exif.get('ExposureTime',''), iso=exif.get('ISOSpeedRatings',''),
                usercomment=str(exif.get('UserComment',''))[:200],
                software=exif.get('Software',''), datetime=exif.get('DateTime','')))

keys=['group','file','width','height','format','bytes','fov_w_mm','fov_h_mm','exif_keys','make','model','focal','fnum','exp','iso','usercomment','software','datetime','error']
with open(os.path.join(OUT,'image_audit.csv'),'w',newline='') as fh:
    wtr=csv.DictWriter(fh,fieldnames=keys); wtr.writeheader()
    for r in rows: wtr.writerow({k:r.get(k,'') for k in keys})

print('total files:',len(rows))
for g in ['categories','OK PHOTOS_MERGED']:
    sel=[r for r in rows if r['group']==g]
    print('\n==',g,len(sel))
    dims=collections.Counter((r.get('width'),r.get('height')) for r in sel)
    for d,c in dims.most_common(10): print('  dim',d,c, 'AR=%.4f'%(d[0]/d[1]) if d[0] else '')
    print('  with FOV in filename:',sum(1 for r in sel if r.get('fov_w_mm')!=''))
    ek=collections.Counter(r.get('exif_keys','') for r in sel)
    for k,c in ek.most_common(5): print('  exifkeys[%d]:'%c, k[:200])
