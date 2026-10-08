"""Post-hoc Grad-CAM for a trained RT-DETR checkpoint. No training, no architecture change.

Explains ONE detection at a time: the target is the pre-sigmoid class logit of a single
decoder query, dec_scores[q, c], read from the same forward pass Ultralytics uses for
prediction. Ultralytics' own detections are the (query, class) pairs whose sigmoid score
clears `conf`, so every detection maps to exactly one (q, c) and its gradient is exact.

Three subcommands, meant to be run in order:

    scan     predict the val split (no CAM), match predictions to ground truth, write
             scan_<tag>.csv with an error category per GT box; also checks that this
             script's forward reproduces model.predict() exactly
    cam      Grad-CAM on a handful of selected GT boxes -> 4-panel figures, a per-layer
             comparison strip, sanity checks, summary.csv/json
    compare  same GT boxes through two checkpoints (e.g. 640 vs 1280) side by side
    full     Grad-CAM on every val GT box -> full_val/full_val.csv, summary.txt, error figures

Error categories (per GT box, detections at conf >= CONF):
    correct               same-class detection with IoU >= 0.5
    classification_error  no same-class match, but a wrong-class detection with IoU >= 0.5
    localization_error    best same-class detection has 0.1 <= IoU < 0.5
    missed_detection      none of the above
For a missed box there is no detection to explain, so the target is the GT-class logit of
the query whose box overlaps the GT most (whatever its score): "what does the model's
nearest query see when asked about the true class".

Grad-CAM (Selvaraju et al.): weights = spatial mean of d(target)/d(A); cam = ReLU(sum_k w_k A_k),
upsampled to the network input and stretched back to the original frame (RT-DETR stretches
rather than letterboxes, so this is the exact inverse).

Border suppression: the outermost ring of feature cells is zeroed before upsampling. Zero
padding in the convolutions produces spurious maxima there (seen on the first run as corner and
edge spikes unrelated to any defect, which hijacked both the pointing game and the colour scale).
The share of CAM energy that sat in that ring is kept as `border_energy_raw` so this is auditable;
pass --keep-border to disable.
"""
import argparse, csv, json, math, os, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PCB = HERE.parent
NAMES = ["Component Liftup", "Component Missing", "Component No Solder", "Component Solder Dry",
         "Polarity Wrong", "RYB Wrong Sequence", "Solder Ball", "Solder Short"]
LAYERS = {21: "P3/8 (fpn_blocks.1)", 24: "P4/16 (pan_blocks.0)", 27: "P5/32 (pan_blocks.1)"}
CONF = 0.25
CATS = ["correct", "localization_error", "classification_error", "missed_detection"]
SMALL_CLASS = "Solder Ball"   # the one small-object class (0.61% median area, OPTIMIZATION_STUDY.md)


# ---------------------------------------------------------------- model + forward
class Explainer:
    def __init__(self, weights, imgsz, device, suppress_border=True):
        from ultralytics import RTDETR
        self.imgsz, self.device, self.suppress_border = imgsz, device, suppress_border
        self.yolo = RTDETR(str(weights))
        self.net = self.yolo.model.to(device).float().eval()
        for p in self.net.parameters():
            p.requires_grad_(False)          # grads only w.r.t. activations
        self.acts, self.hooks = {}, []
        for i in LAYERS:
            self.hooks.append(self.net.model[i].register_forward_hook(self._save(i)))

    def _save(self, i):
        def fn(_m, _inp, out):
            if out.requires_grad:            # skip no-grad passes (e.g. the predict() reference)
                out.retain_grad()
                self.acts[i] = out
        return fn

    def preprocess(self, bgr):
        im = cv2.resize(bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)  # = LetterBox(scale_fill)
        x = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float().div(255)
        return x[None].to(self.device)

    def forward(self, bgr):
        """Returns (boxes_xyxy_norm[nq,4], logits[nq,nc]) with the autograd graph kept."""
        x = self.preprocess(bgr).requires_grad_(True)   # makes the graph exist without param grads
        with torch.enable_grad():
            _y, (dec_bboxes, dec_scores, *_rest) = self.net(x)
        b = dec_bboxes[-1, 0]                            # (nq, 4) cxcywh in [0,1]
        boxes = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
        return boxes, dec_scores[-1, 0]

    @staticmethod
    def detections(boxes, logits, W, H, conf=CONF):
        """Reproduces RTDETRDecoder.postprocess + RTDETRPredictor: top-k over (query x class)."""
        nq, nc = logits.shape
        s = logits.detach().sigmoid().flatten()
        sc, idx = s.topk(nq)
        out = []
        for v, i in zip(sc.tolist(), idx.tolist()):
            if v <= conf:
                continue
            q, c = divmod(i, nc)
            bx = boxes[q].detach().cpu().numpy() * [W, H, W, H]
            out.append(dict(query=q, cls=c, conf=v, box=bx.tolist()))
        return out

    def gradcam(self, logits, q, c, size_wh):
        """CAM for target logits[q, c] at every hooked layer -> {layer: (cam[H,W] in [0,1], stats)}."""
        for a in self.acts.values():
            a.grad = None
        logits[q, c].backward(retain_graph=True)
        W, H = size_wh
        res = {}
        for i, A in self.acts.items():
            G = A.grad
            stats = dict(grad_abs_mean=float(G.abs().mean()), grad_finite=bool(torch.isfinite(G).all()),
                         act_finite=bool(torch.isfinite(A).all()), fmap=f"{A.shape[-1]}x{A.shape[-2]}")
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * A).sum(1, keepdim=True))
            inner = torch.zeros_like(cam); inner[..., 1:-1, 1:-1] = 1
            stats["border_energy_raw"] = float((cam * (1 - inner)).sum() / (cam.sum() + 1e-12))
            if self.suppress_border:
                cam = cam * inner
            cam = F.interpolate(cam, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
            stats["cam_raw_max"] = float(cam.max())
            stats["cam_finite"] = bool(np.isfinite(cam).all())
            cam = cam / cam.max() if cam.max() > 0 else cam
            res[i] = (cam, stats)
        return res


# ---------------------------------------------------------------- data + matching
def load_val(data_root):
    img_dir, lab_dir = data_root / "images/val", data_root / "labels/val"
    items = []
    for p in sorted(img_dir.iterdir()):
        lp = lab_dir / (p.stem + ".txt")
        gts = []
        if lp.exists():
            for ln in lp.read_text().split("\n"):
                if ln.strip():
                    c, cx, cy, w, h = ln.split()
                    gts.append((int(c), float(cx), float(cy), float(w), float(h)))
        items.append((p, gts))
    return items


def gt_xyxy(g, W, H):
    _, cx, cy, w, h = g
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def categorize(gc, gbox, dets, boxes_all, W, H):
    """-> (category, target dict). Target is the detection (or query) the CAM will explain."""
    same = sorted([(iou(gbox, d["box"]), d) for d in dets if d["cls"] == gc], key=lambda t: -t[0])
    other = sorted([(iou(gbox, d["box"]), d) for d in dets if d["cls"] != gc], key=lambda t: -t[0])
    if same and same[0][0] >= 0.5:
        return "correct", dict(same[0][1], iou=same[0][0], kind="detection")
    if other and other[0][0] >= 0.5:
        return "classification_error", dict(other[0][1], iou=other[0][0], kind="detection")
    if same and same[0][0] >= 0.1:
        return "localization_error", dict(same[0][1], iou=same[0][0], kind="detection")
    qb = boxes_all.detach().cpu().numpy() * [W, H, W, H]
    ious = [iou(gbox, b.tolist()) for b in qb]
    q = int(np.argmax(ious))
    return "missed_detection", dict(query=q, cls=gc, conf=None, box=qb[q].tolist(), iou=ious[q],
                                    kind="nearest_query_gt_class")


# ---------------------------------------------------------------- figures
def draw_box(ax, b, color, label=None, ls="-"):
    import matplotlib.patches as P
    ax.add_patch(P.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=2, ls=ls))
    if label:
        ax.text(b[0], max(b[1] - 6, 10), label, color="white", fontsize=8,
                bbox=dict(facecolor=color, alpha=0.85, pad=1.5, lw=0))


def four_panel(rgb, gbox, gname, tgt, cam, layer, title, dest):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(22, 3.9))
    for a in ax:
        a.axis("off")
    ax[0].imshow(rgb); ax[0].set_title("1  Original")
    ax[1].imshow(rgb); ax[1].set_title("2  Ground truth (green) + prediction (red)")
    draw_box(ax[1], gbox, "#1a9850", f"GT: {gname}")
    plabel = (f"{NAMES[tgt['cls']]} {tgt['conf']:.2f} | IoU {tgt['iou']:.2f} | q{tgt['query']}"
              if tgt["kind"] == "detection"
              else f"no det >= {CONF}; nearest q{tgt['query']} IoU {tgt['iou']:.2f}")
    draw_box(ax[1], tgt["box"], "#d73027", plabel, ls="-" if tgt["kind"] == "detection" else "--")
    ax[2].imshow(cam, cmap="jet", vmin=0, vmax=1); ax[2].set_title(f"3  Grad-CAM, layer {layer} {LAYERS[layer].split()[0]}")
    ax[3].imshow(rgb); ax[3].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax[3].set_title("4  Overlay")
    draw_box(ax[3], gbox, "#1a9850"); draw_box(ax[3], tgt["box"], "#d73027", ls="-" if tgt["kind"] == "detection" else "--")
    fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout(); dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110, bbox_inches="tight"); plt.close(fig)


def layer_strip(rgb, gbox, tbox, cams, title, dest):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(cams), figsize=(5.5 * len(cams), 3.6))
    for a, (L, (cam, st)) in zip(ax, cams.items()):
        a.imshow(rgb); a.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1); a.axis("off")
        draw_box(a, gbox, "#1a9850"); draw_box(a, tbox, "#d73027")
        a.set_title(f"layer {L} {LAYERS[L]}  [{st['fmap']}]", fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=90, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- CAM focus metrics
def box_mask(b, W, H):
    m = np.zeros((H, W), bool)
    x1, y1, x2, y2 = [int(round(v)) for v in (max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H))]
    m[y1:y2, x1:x2] = True
    return m


def focus(cam, gbox, tbox, W, H):
    """Energy fraction inside a box vs. that box's share of the frame (>1 = concentrated there)."""
    gm, tm = box_mask(gbox, W, H), box_mask(tbox, W, H)
    tot = cam.sum() + 1e-12
    py, px = np.unravel_index(cam.argmax(), cam.shape)
    return dict(
        energy_in_gt=float(cam[gm].sum() / tot), gt_area_frac=float(gm.mean()),
        energy_in_pred=float(cam[tm].sum() / tot), pred_area_frac=float(tm.mean()),
        peak_in_gt=bool(gm[py, px]), peak_in_pred=bool(tm[py, px]),
        spread=float((cam > 0.5).mean()),       # share of frame above half-max: high = diffuse
    )


def describe(f):
    lift = f["energy_in_gt"] / max(f["gt_area_frac"], 1e-6)
    if f["peak_in_gt"] and lift >= 2:
        return f"on defect (peak in GT, {f['energy_in_gt']:.0%} energy in {f['gt_area_frac']:.0%} of frame)"
    if f["peak_in_gt"]:
        return f"peak on defect but spread ({f['spread']:.0%} of frame > half-max)"
    if f["peak_in_pred"]:
        return f"on predicted box, off GT ({f['energy_in_gt']:.0%} energy in GT)"
    if lift >= 1.5:
        return f"near defect, peak outside GT ({f['energy_in_gt']:.0%} energy in GT)"
    return f"off defect: background ({f['energy_in_gt']:.0%} energy in GT, {f['gt_area_frac']:.0%} of frame)"


# ---------------------------------------------------------------- subcommands
def cmd_scan(a):
    ex = Explainer(a.weights, a.imgsz, a.device)
    from ultralytics import RTDETR
    ref_model = RTDETR(str(a.weights))   # separate instance: predict() fuses layers in place
    items = load_val(a.data)
    rows, check = [], []
    for k, (p, gts) in enumerate(items):
        bgr = cv2.imread(str(p)); H, W = bgr.shape[:2]
        boxes, logits = ex.forward(bgr)
        dets = ex.detections(boxes, logits, W, H)
        if k < a.verify:   # this forward must equal Ultralytics' own predictor
            r = ref_model.predict(str(p), imgsz=a.imgsz, conf=CONF, device=a.device, verbose=False)[0]
            ref = sorted(zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda t: -t[1])
            mine = sorted([(d["cls"], d["conf"], d["box"]) for d in dets], key=lambda t: -t[1])
            ok = len(ref) == len(mine) and all(c1 == c2 and abs(s1 - s2) < 5e-3 and iou(b1, b2) > 0.99
                                                for (c1, s1, b1), (c2, s2, b2) in zip(ref, mine))
            check.append(dict(image=p.name, n_ref=len(ref), n_mine=len(mine), match=ok))
        for gi, g in enumerate(gts):
            gb = gt_xyxy(g, W, H)
            cat, t = categorize(g[0], gb, dets, boxes, W, H)
            rows.append(dict(image=p.name, gt_idx=gi, ground_truth_class=NAMES[g[0]],
                             gt_w_px=round(gb[2] - gb[0], 1), gt_h_px=round(gb[3] - gb[1], 1),
                             predicted_class=NAMES[t["cls"]] if t["kind"] == "detection" else "",
                             confidence=round(t["conf"], 4) if t["conf"] is not None else "",
                             IoU=round(t["iou"], 4), error_category=cat, target_query=t["query"],
                             n_dets=len(dets)))
        if k % 25 == 0:
            print(f"[scan] {k}/{len(items)}", flush=True)
    out = HERE / f"scan_{a.tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (HERE / f"scan_{a.tag}_predict_check.json").write_text(json.dumps(check, indent=1))
    from collections import Counter
    print("[scan] categories:", dict(Counter(r["error_category"] for r in rows)))
    print(f"[scan] forward == model.predict on {sum(c['match'] for c in check)}/{len(check)} images")
    print("[scan] ->", out)


def pick(rows, n_per=2):
    """Deterministic selection covering every category + the small-object class + Liftup."""
    chosen, seen = [], set()
    def take(pred, n, why):
        for r in sorted(filter(pred, rows), key=lambda r: (r["image"], int(r["gt_idx"]))):
            key = (r["image"], r["gt_idx"])
            if key in seen or n == 0:
                continue
            seen.add(key); chosen.append(dict(r, why=why)); n -= 1
    for c in CATS:
        take(lambda r, c=c: r["error_category"] == c and r["ground_truth_class"] != SMALL_CLASS, n_per, c)
    take(lambda r: r["ground_truth_class"] == SMALL_CLASS and r["error_category"] == "correct", 1, "small_objects")
    take(lambda r: r["ground_truth_class"] == SMALL_CLASS and r["error_category"] != "correct", 1, "small_objects")
    take(lambda r: r["ground_truth_class"] == "Component Liftup", 1, "component_liftup")
    return chosen


def run_cam_rows(ex, a, rows, alt_check=True):
    out = []
    for r in rows:
        p = a.data / "images/val" / r["image"]
        bgr = cv2.imread(str(p)); H, W = bgr.shape[:2]; rgb = bgr[..., ::-1]
        g = load_label(a.data, p)[int(r["gt_idx"])]
        gb = gt_xyxy(g, W, H)
        boxes, logits = ex.forward(bgr)
        dets = ex.detections(boxes, logits, W, H)
        cat, t = categorize(g[0], gb, dets, boxes, W, H)
        cams = ex.gradcam(logits, t["query"], t["cls"], (W, H))
        # sanity: a *different* target on the same image must give a different map
        alt_q = next((d["query"] for d in dets if d["query"] != t["query"]), (t["query"] + 1) % logits.shape[0])
        alt_c = int(logits[alt_q].argmax())
        alt = ex.gradcam(logits, alt_q, alt_c, (W, H)) if alt_check else None
        out.append(dict(r=r, p=p, rgb=rgb, W=W, H=H, gb=gb, g=g, cat=cat, t=t, cams=cams, alt=alt,
                        alt_target=(alt_q, alt_c), logit=float(logits[t["query"], t["cls"]])))
    return out


def load_label(data, p):
    return [tuple([int(x.split()[0])] + [float(v) for v in x.split()[1:]])
            for x in (data / "labels/val" / (p.stem + ".txt")).read_text().split("\n") if x.strip()]


def cmd_cam(a):
    scan = list(csv.DictReader(open(HERE / f"scan_{a.tag}.csv")))
    rows = pick(scan)
    ex = Explainer(a.weights, a.imgsz, a.device, not a.keep_border)
    res = run_cam_rows(ex, a, rows)

    # Layer choice on a held-out set of correct detections (not the figure set), by the
    # pointing game: does the CAM peak land inside the GT box? Energy-in-box is reported as a
    # tie-break only -- it rewards whole-frame noise when the GT box is large.
    used = {(r["image"], r["gt_idx"]) for r in rows}
    sel = [dict(r, why="layer_selection") for r in sorted(scan, key=lambda r: (r["image"], int(r["gt_idx"])))
           if r["error_category"] == "correct" and (r["image"], r["gt_idx"]) not in used][::7][: a.n_select]
    sel_res = run_cam_rows(ex, a, sel)
    layer_score = {}
    for L in LAYERS:
        fs = [focus(x["cams"][L][0], x["gb"], x["t"]["box"], x["W"], x["H"]) for x in sel_res]
        layer_score[L] = dict(pointing_game=float(np.mean([f["peak_in_gt"] for f in fs])),
                              mean_lift=float(np.mean([f["energy_in_gt"] / max(f["gt_area_frac"], 1e-6) for f in fs])),
                              mean_spread=float(np.mean([f["spread"] for f in fs])))
    best = a.layer or max(LAYERS, key=lambda L: (layer_score[L]["pointing_game"], layer_score[L]["mean_lift"]))
    print(f"[cam] layer selection on {len(sel_res)} held-out correct detections:", layer_score, "-> layer", best)

    summary = []
    for x in res:
        r, t = x["r"], x["t"]
        folder = r["why"] if r["why"] in ("small_objects",) else x["cat"]
        stem = f"{Path(r['image']).stem}_gt{r['gt_idx']}"
        cam, st = x["cams"][best]
        f = focus(cam, x["gb"], t["box"], x["W"], x["H"])
        corr = float(np.corrcoef(cam.ravel(), x["alt"][best][0].ravel())[0, 1])
        title = (f"{r['image']}  |  {x['cat']}  |  target: query {t['query']} class '{NAMES[t['cls']]}' "
                 f"logit {x['logit']:.2f}  |  CAM: {describe(f)}")
        four_panel(x["rgb"], x["gb"], NAMES[x["g"][0]], t, cam, best, title, HERE / folder / f"{stem}.png")
        layer_strip(x["rgb"], x["gb"], t["box"], x["cams"], f"{r['image']} — {x['cat']} — layer comparison",
                    HERE / "layer_comparison" / f"{stem}.png")
        summary.append(dict(
            image=r["image"], gt_idx=r["gt_idx"], folder=folder,
            ground_truth_class=NAMES[x["g"][0]],
            predicted_class=NAMES[t["cls"]] if t["kind"] == "detection" else "(none)",
            confidence=round(t["conf"], 3) if t["conf"] is not None else None,
            IoU=round(t["iou"], 3), error_category=x["cat"], target_layer=best,
            target_prediction=(f"query {t['query']}, class {NAMES[t['cls']]}, logit {x['logit']:.3f}"
                               + ("" if t["kind"] == "detection" else " (nearest query, GT class; no detection)")),
            cam_focus=describe(f), **{k: round(v, 4) if isinstance(v, float) else v for k, v in f.items()},
            gt_size_px=f"{r['gt_w_px']}x{r['gt_h_px']}",
            border_energy_raw=round(st["border_energy_raw"], 4),
            sanity_grad_abs_mean=st["grad_abs_mean"], sanity_finite=st["grad_finite"] and st["cam_finite"] and st["act_finite"],
            sanity_corr_vs_other_target=round(corr, 3), other_target=f"q{x['alt_target'][0]} {NAMES[x['alt_target'][1]]}",
            **{f"peak_in_gt_layer{L}": focus(x['cams'][L][0], x['gb'], t['box'], x['W'], x['H'])['peak_in_gt'] for L in LAYERS},
        ))
    # cross-image distinctness: CAMs of different images resized to a common grid
    small = [cv2.resize(x["cams"][best][0], (64, 36)).ravel() for x in res]
    cc = np.corrcoef(np.stack(small)); iu = np.triu_indices(len(small), 1)
    meta = dict(weights=str(a.weights), imgsz=a.imgsz, conf=CONF, selected_layer=best, border_suppressed=not a.keep_border,
                layer_selection_metric="pointing game (CAM peak inside GT box) on held-out correct detections; tie-break mean energy lift",
                layer_selection_images=[x["r"]["image"] for x in sel_res],
                layer_scores=layer_score, cross_image_cam_corr_mean=float(cc[iu].mean()),
                cross_image_cam_corr_max=float(cc[iu].max()), n=len(summary))
    with open(HERE / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (HERE / "summary.json").write_text(json.dumps(dict(meta=meta, rows=summary), indent=1))
    print(json.dumps(meta, indent=1))


def cmd_full(a):
    """Grad-CAM on EVERY val GT box at one layer -> full_val.csv + plain-count summary.
    Figures only for the errors (correct ones would be ~80 near-identical pictures)."""
    from collections import Counter, defaultdict
    scan = list(csv.DictReader(open(HERE / f"scan_{a.tag}.csv")))
    layer = a.layer or 24
    ex = Explainer(a.weights, a.imgsz, a.device, not a.keep_border)
    out = []
    for k, x in enumerate(run_cam_rows(ex, a, [dict(r, why="full") for r in scan], alt_check=False)):
        r, t = x["r"], x["t"]
        cam = x["cams"][layer][0]
        f = focus(cam, x["gb"], t["box"], x["W"], x["H"])
        cat = x["cat"]
        if cat == "missed_detection":   # did any query land on the defect at all?
            cat_detail = "looked_but_low_score" if t["iou"] >= 0.5 else "never_looked"
        else:
            cat_detail = cat
        out.append(dict(image=r["image"], gt_idx=r["gt_idx"], ground_truth_class=NAMES[x["g"][0]],
                        error_category=cat, detail=cat_detail,
                        predicted_class=NAMES[t["cls"]] if t["kind"] == "detection" else "",
                        confidence=round(t["conf"], 3) if t["conf"] is not None else "",
                        gt_class_prob=round(1 / (1 + math.exp(-x["logit"])), 4) if t["kind"] != "detection" else "",
                        IoU=round(t["iou"], 3), gt_size_px=f"{r['gt_w_px']}x{r['gt_h_px']}",
                        gt_area_frac=round(f["gt_area_frac"], 4),
                        cam_peak_on_defect=f["peak_in_gt"],
                        cam_energy_lift=round(f["energy_in_gt"] / max(f["gt_area_frac"], 1e-6), 2),
                        cam_spread=round(f["spread"], 3)))
        if cat != "correct":
            stem = f"{Path(r['image']).stem}_gt{r['gt_idx']}"
            four_panel(x["rgb"], x["gb"], NAMES[x["g"][0]], t, cam, layer,
                       f"{r['image']}  |  {cat_detail}  |  CAM: {describe(f)}",
                       HERE / "full_val" / cat / f"{stem}.png")
        if k % 20 == 0:
            print(f"[full] {k}/{len(scan)}", flush=True)
    d = HERE / "full_val"; d.mkdir(exist_ok=True)
    with open(d / "full_val.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    # ---- plain summary
    L = []
    n = len(out)
    L.append(f"Grad-CAM on all {n} validation defects (layer {layer}, conf {CONF})\n")
    L.append("1. What happened to each defect")
    for c, v in Counter(o["detail"] for o in out).most_common():
        L.append(f"   {c:24s} {v:4d}  ({v / n:.0%})")
    L.append("\n2. Was the heatmap peak on the defect?")
    by = defaultdict(list)
    for o in out:
        by[o["detail"]].append(o["cam_peak_on_defect"])
    for c, v in by.items():
        L.append(f"   {c:24s} {sum(v):3d}/{len(v):<3d} ({sum(v) / len(v):.0%})")
    L.append("\n3. Per class")
    L.append(f"   {'class':22s} {'n':>3s} {'correct':>8s} {'missed':>7s} {'wrong cls':>9s} {'bad box':>8s} {'CAM on defect':>14s}")
    for cname in NAMES:
        rows = [o for o in out if o["ground_truth_class"] == cname]
        if not rows:
            continue
        cc = Counter(o["error_category"] for o in rows)
        on = sum(o["cam_peak_on_defect"] for o in rows)
        L.append(f"   {cname:22s} {len(rows):3d} {cc['correct']:8d} {cc['missed_detection']:7d} "
                 f"{cc['classification_error']:9d} {cc['localization_error']:8d} {on:8d}/{len(rows):<3d}")
    L.append("\n4. Wrong-class detections (true -> predicted)")
    for (tc, pc), v in Counter((o["ground_truth_class"], o["predicted_class"]) for o in out
                               if o["error_category"] == "classification_error").most_common():
        L.append(f"   {tc} -> {pc}: {v}")
    L.append("\n5. Defect size: correct vs missed (median share of frame)")
    for c in ("correct", "missed_detection"):
        v = sorted(o["gt_area_frac"] for o in out if o["error_category"] == c)
        if v:
            L.append(f"   {c:24s} {v[len(v) // 2]:.2%}  (n={len(v)})")
    txt = "\n".join(L)
    (d / "summary.txt").write_text(txt + "\n")
    print(txt)


def cmd_compare(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = json.loads((HERE / "summary.json").read_text()); rows = S["rows"]
    layer = a.layer or S["meta"]["selected_layer"]
    # small-object (Solder Ball) cases first -- they are the point of a resolution comparison
    rows = sorted([r for r in rows if r["ground_truth_class"] == SMALL_CLASS or r["error_category"] != "correct"],
                  key=lambda r: r["ground_truth_class"] != SMALL_CLASS)[: a.n]
    ea = Explainer(a.weights, a.imgsz, a.device, not a.keep_border)
    eb = Explainer(a.weights_b, a.imgsz_b, a.device, not a.keep_border)
    out = []
    for r in rows:
        p = a.data / "images/val" / r["image"]; bgr = cv2.imread(str(p)); H, W = bgr.shape[:2]; rgb = bgr[..., ::-1]
        g = load_label(a.data, p)[int(r["gt_idx"])]; gb = gt_xyxy(g, W, H)
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); line = dict(image=r["image"], gt=NAMES[g[0]])
        for a_, (ex, tag) in zip(ax, ((ea, f"{a.imgsz}px"), (eb, f"{a.imgsz_b}px"))):
            boxes, logits = ex.forward(bgr); dets = ex.detections(boxes, logits, W, H)
            cat, t = categorize(g[0], gb, dets, boxes, W, H)
            cam = ex.gradcam(logits, t["query"], t["cls"], (W, H))[layer][0]
            f = focus(cam, gb, t["box"], W, H)
            a_.imshow(rgb); a_.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1); a_.axis("off")
            draw_box(a_, gb, "#1a9850"); draw_box(a_, t["box"], "#d73027", ls="-" if t["kind"] == "detection" else "--")
            conf = f"{t['conf']:.2f}" if t["conf"] is not None else "-"
            a_.set_title(f"{tag}: {cat} | {NAMES[t['cls']]} conf {conf} IoU {t['iou']:.2f}\nCAM: {describe(f)}", fontsize=8)
            line.update({f"{tag}_category": cat, f"{tag}_conf": conf, f"{tag}_iou": round(t["iou"], 3),
                         f"{tag}_energy_in_gt": round(f["energy_in_gt"], 3), f"{tag}_peak_in_gt": f["peak_in_gt"]})
        fig.tight_layout(); d = HERE / "res_640_vs_1280" / f"{Path(r['image']).stem}_gt{r['gt_idx']}.png"
        d.parent.mkdir(exist_ok=True); fig.savefig(d, dpi=100, bbox_inches="tight"); plt.close(fig)
        out.append(line)
    (HERE / "res_640_vs_1280" / "compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "cam", "compare", "full"])
    ap.add_argument("--weights", type=Path, default=PCB / "runs/experiments/nq30/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", type=Path, default=PCB / "scratch/exp_data/neg120")   # nq30's own val split
    ap.add_argument("--tag", default="nq30")
    ap.add_argument("--layer", type=int, choices=list(LAYERS), help="override automatic layer choice")
    ap.add_argument("--verify", type=int, default=15, help="images checked against model.predict in scan")
    ap.add_argument("--weights-b", type=Path, default=PCB / "runs/experiments/nq30_r1280/weights/best.pt")
    ap.add_argument("--imgsz-b", type=int, default=1280)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--keep-border", action="store_true", help="disable border-cell suppression")
    ap.add_argument("--n-select", type=int, default=10, help="held-out correct detections used to pick the layer")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    {"scan": cmd_scan, "cam": cmd_cam, "compare": cmd_compare, "full": cmd_full}[a.cmd](a)


if __name__ == "__main__":
    main()
