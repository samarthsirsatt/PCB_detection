"""Shared data pipeline for the RT-DETR optimisation experiments.

Reuses rtdetr.py's exact COCO->YOLO logic and stratified split (SEED=42) so every
experiment is comparable to the published 0.770 baseline. The ONLY things an
experiment may vary are exposed as env vars in exp_rtdetr.py.

Datasets are cached per (n_negatives, fold) under $SCRATCH_ROOT/exp_data/ so
parallel array tasks sharing a config don't rebuild ~1,700 small files each.
"""
import json, os, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

from hpc_env import COCO_JSON, IMG_ROOT, NEG_ROOT, SCRATCH_ROOT

SEED = 42
VAL_FRAC = 0.20
KEEP = ["Component Liftup", "Component Missing", "Component No Solder",
        "Component Solder Dry", "Polarity Wrong", "RYB Wrong Sequence",
        "Solder Ball", "Solder Short"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def coco_bbox_to_yolo(b, W, H):
    x, y, w, h = b
    return (x + w / 2) / W, (y + h / 2) / H, w / W, h / H


def filter_coco_to_classes(coco, keep_names):
    keep_set = set(keep_names)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    new_cats = [{"id": i + 1, "name": n} for i, n in enumerate(keep_names)]
    name2new = {c["name"]: c["id"] for c in new_cats}
    new_anns = []
    for a in coco["annotations"]:
        nm = id2name.get(a["category_id"])
        if nm in keep_set:
            b = dict(a); b["category_id"] = name2new[nm]; new_anns.append(b)
    imgs_with_ann = {a["image_id"] for a in new_anns}
    return {"images": [im for im in coco["images"] if im["id"] in imgs_with_ann],
            "annotations": new_anns, "categories": new_cats}


def load_8class():
    coco = json.loads(Path(COCO_JSON).read_text())
    src = {}
    base = Path(IMG_ROOT).parent
    for p in Path(IMG_ROOT).rglob("*"):
        if p.suffix.lower() in EXTS:
            src[str(p.relative_to(base))] = p
    bidx = defaultdict(list)
    for k, v in src.items():
        bidx[Path(k).name].append(v)

    def resolve(fn):
        if fn in src:
            return src[fn]
        h = bidx.get(Path(fn).name, [])
        return h[0] if len(h) == 1 else None

    found = sum(1 for im in coco["images"] if resolve(im["file_name"]))
    assert found == len(coco["images"]), \
        f"image resolution incomplete: {found}/{len(coco['images'])} — see SITE_AMENDMENTS A9"
    return filter_coco_to_classes(coco, KEEP), resolve


def stratified_split(coco8):
    """rtdetr.py's split, verbatim: stratify each image by its RAREST class."""
    freq = Counter(a["category_id"] for a in coco8["annotations"])
    img_cls = defaultdict(set)
    for a in coco8["annotations"]:
        img_cls[a["image_id"]].add(a["category_id"])
    strat = defaultdict(list)
    for im in coco8["images"]:
        cls = img_cls.get(im["id"], set())
        key = min(cls, key=lambda c: freq[c]) if cls else "__none__"
        strat[key].append(im["id"])
    rng = random.Random(SEED)
    tr, va = set(), set()
    for _, ids in strat.items():
        ids = ids[:]; rng.shuffle(ids)
        nv = max(1, round(len(ids) * VAL_FRAC)) if len(ids) > 1 else 0
        va |= set(ids[:nv]); tr |= set(ids[nv:])
    assert not (tr & va)
    return tr, va


def stratified_kfold(coco8, k=5, seed=SEED):
    """Round-robin folds within each rarest-class stratum (same key as the split)."""
    freq = Counter(a["category_id"] for a in coco8["annotations"])
    img_cls = defaultdict(set)
    for a in coco8["annotations"]:
        img_cls[a["image_id"]].add(a["category_id"])
    strat = defaultdict(list)
    for im in coco8["images"]:
        cls = img_cls.get(im["id"], set())
        key = min(cls, key=lambda c: freq[c]) if cls else "__none__"
        strat[key].append(im["id"])
    rng = random.Random(seed)
    fold_of = {}
    for _, ids in strat.items():
        ids = ids[:]; rng.shuffle(ids)
        for j, i in enumerate(ids):
            fold_of[i] = j % k
    return fold_of


def _write_split(coco, ids, split, out, resolve):
    (out / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (out / f"labels/{split}").mkdir(parents=True, exist_ok=True)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cid2yolo = {c["id"]: i for i, c in enumerate(cats)}
    byid = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    ni = nb = 0
    for i in ids:
        im = byid[i]; s = resolve(im["file_name"])
        if s is None:
            continue
        uniq = im["file_name"].replace("/", "__").replace("\\", "__")
        shutil.copy(s, out / f"images/{split}/{uniq}")
        W, H = im["width"], im["height"]
        lines = []
        for a in anns.get(i, []):
            cx, cy, w, h = coco_bbox_to_yolo(a["bbox"], W, H)
            lines.append(f"{cid2yolo[a['category_id']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            nb += 1
        (out / f"labels/{split}/{Path(uniq).stem}.txt").write_text("\n".join(lines))
        ni += 1
    return ni, nb


def build_dataset(n_neg=120, fold=None, force=False):
    """Build (or reuse) a YOLO tree. fold=None -> the standard 80/20 split."""
    tag = f"neg{n_neg}" + (f"_fold{fold}" if fold is not None else "")
    out = Path(SCRATCH_ROOT) / "exp_data" / tag
    yaml_p = out / "data.yaml"
    if yaml_p.exists() and not force:
        print(f"[data] reusing {out}")
        return out
    print(f"[data] building {out}")
    shutil.rmtree(out, ignore_errors=True)

    coco8, resolve = load_8class()
    names = [c["name"] for c in sorted(coco8["categories"], key=lambda c: c["id"])]

    if fold is None:
        tr, va = stratified_split(coco8)
    else:
        fo = stratified_kfold(coco8, 5)
        allids = [im["id"] for im in coco8["images"]]
        va = {i for i in allids if fo.get(i) == fold}
        tr = {i for i in allids if fo.get(i) != fold}

    ni_t, nb_t = _write_split(coco8, tr, "train", out, resolve)
    ni_v, nb_v = _write_split(coco8, va, "val", out, resolve)
    print(f"[data] pos train {ni_t} imgs/{nb_t} boxes | val {ni_v} imgs/{nb_v} boxes")

    negs = [p for p in Path(NEG_ROOT).rglob("*") if p.suffix.lower() in EXTS]
    random.Random(SEED).shuffle(negs)
    negs = negs[:n_neg]
    nval = round(len(negs) * VAL_FRAC)
    for split, chunk in (("val", negs[:nval]), ("train", negs[nval:])):
        for p in chunk:
            shutil.copy(p, out / f"images/{split}/{p.name}")
            (out / f"labels/{split}/{p.stem}.txt").write_text("")
    print(f"[data] negatives: {len(negs) - nval} train, {nval} val (of {n_neg} requested)")

    import yaml
    yaml_p.write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/train", "val": "images/val",
         "names": {i: n for i, n in enumerate(names)}}, sort_keys=False))
    return out


def make_nq_yaml(nq, dest):
    """Copy ultralytics' rtdetr-l.yaml with RTDETRDecoder args -> [nc, hd=256, nq].

    parse_model inserts `ch` at index 1, so YAML args map to (nc, ch, hd, nq).
    """
    import ultralytics
    src = Path(ultralytics.__file__).parent / "cfg/models/rt-detr/rtdetr-l.yaml"
    txt = src.read_text()
    old = "[[21, 24, 27], 1, RTDETRDecoder, [nc]]"
    new = f"[[21, 24, 27], 1, RTDETRDecoder, [nc, 256, {nq}]]"
    assert old in txt, "decoder line not found — ultralytics layout changed"
    dest = Path(dest)
    # Atomic: parallel nq jobs share this path, so never expose a half-written file.
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(txt.replace(old, new))
    os.replace(tmp, dest)
    return dest
