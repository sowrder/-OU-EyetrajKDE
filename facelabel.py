#!/usr/bin/env python3
"""
facelabel.py — stage 2 of the segmentation pipeline: the face regions Q_v.

ROLE IN THE DOCUMENT
  Supplies, for every stimulus image, one ellipse per detected face.  These
  are the Q_v of §4.2 against which basin centres μ_k are tested.  They are
  produced from the images alone and never enter the estimation of p̂ or of
  the basins, which is what keeps the face results non-circular.

THE ELLIPSE
  68-point landmarks from face_alignment.  The ellipse is the second-moment
  fit of the 68 points: centre at their mean, axes along the eigenvectors of
  their covariance, semi-axes 2 standard deviations, scaled by 1.06.  In the
  face's own rotated frame (u, v),

      Q_v(x) = (u / a_v)² + (v / b_v)²,     inside when Q_v ≤ 1.

  The centre is the landmark centroid, which lies near the nose rather than
  the eyes; the depth q = min_v Q_v(μ_k) of §4.3 is measured from it.

  A polygon (jaw arc plus a lifted brow arc) is also written.  It is not used
  in the document.

COORDINATES
  Landmarks are in image pixels.  Stimuli are 1024×768 centred on the
  1280×1024 screen, so screen = image + (128, 128).  Gaze is in screen
  coordinates, so cx_scr, cy_scr are the columns used downstream.

SIZE FLAG
  Detections outside the 2.5–97.5 percentiles of bbox area are flagged, not
  dropped.  The document uses every detection.

OUTPUT (--outdir)
  face_bounds.csv        one row per face: centre, a_v, b_v, angle, area
  face_polygons.csv      polygon vertices (unused)
  face_landmarks.csv     all 68 points per face
  overlay/<stem>.png     image with landmarks and ellipse, for inspection
"""
from __future__ import annotations
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE','1')
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

OFFSET = np.array([128.0, 128.0])     # image -> screen
FOREHEAD = 0.42                       # brow lift, x eye-to-chin distance
ELLIPSE = 1.06                        # scale on the fitted ellipse
SIZE_LO, SIZE_HI = 2.5, 97.5

JAW = list(range(0, 17))
BROW = list(range(17, 27))
EYE_L = list(range(36, 42))
EYE_R = list(range(42, 48))
NOSE_TIP = 33
CHIN = 8


def fit_ellipse(P: np.ndarray):
    """second-moment ellipse of the landmark cloud: centre, semi-axes a_v, b_v
    and angle.  Chosen over a general conic fit because it cannot return a
    hyperbola when the points are noisy."""
    c = P.mean(0)
    Q = P - c
    C = (Q.T @ Q) / len(Q)
    ev, evec = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    # semi-axes at 2 sd, which encloses a uniform elliptical cloud; the
    # 1.06 scale is applied by the caller
    a, b = 2.0 * np.sqrt(np.maximum(ev, 1e-9))
    ang = float(np.degrees(np.arctan2(evec[1, 0], evec[0, 0])))
    return c, float(a), float(b), ang


def build_polygon(lm: np.ndarray) -> np.ndarray:
    """jaw arc plus a lifted brow arc, closed.  Written, not used."""
    jaw = lm[JAW]
    brow = lm[BROW]
    chin = lm[CHIN]
    eye = np.vstack([lm[EYE_L], lm[EYE_R]]).mean(0)
    h = np.linalg.norm(chin - eye)
    up = eye - chin
    up = up / (np.linalg.norm(up) + 1e-9)
    fore = brow + up * (FOREHEAD * h)
    return np.vstack([jaw, fore[::-1]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", required=True,
                    help="directory of images, e.g. .../ImagesVault/IAPS")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--images", default="", help="comma-separated stems")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()

    out = Path(a.outdir); (out/"overlay").mkdir(parents=True, exist_ok=True)
    files = sorted(sum((glob.glob(f"{a.imgdir}/*{e}")
                        for e in (".jpg", ".jpeg", ".png", ".bmp")), []))
    want = [w.strip() for w in a.images.split(",") if w.strip()]
    if want:
        files = [f for f in files if Path(f).stem in want]
    if not files:
        raise FileNotFoundError(f"no images under {a.imgdir}")
    print(f"{len(files)} images", flush=True)

    import torch
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if dev != a.device:
        print(f"  cuda unavailable — falling back to cpu")
    print(f"  device: {dev}"
          + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""),
          flush=True)
    import face_alignment
    from PIL import Image
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device=dev)

    rows, polys, lms = [], [], []
    for n, f in enumerate(files, 1):
        stem = Path(f).stem
        im = np.array(Image.open(f).convert("RGB"))
        try:
            preds = fa.get_landmarks(im)
        except Exception as e:
            print(f"  [{stem}] detect failed {type(e).__name__}"); continue
        if not preds:
            print(f"  [{stem}] no face"); continue
        for fid, lm in enumerate(preds, 1):
            lm = np.asarray(lm, float)[:, :2]
            poly = build_polygon(lm)
            c, ax_a, ax_b, ang = fit_ellipse(lm)
            ax_a *= ELLIPSE; ax_b *= ELLIPSE
            x0, y0 = lm.min(0); x1, y1 = lm.max(0)
            rows.append(dict(
                image=Path(f).name, stem=stem, face_id=fid,
                cx_img=c[0], cy_img=c[1],
                cx_scr=c[0]+OFFSET[0], cy_scr=c[1]+OFFSET[1],
                ell_a=ax_a, ell_b=ax_b, ell_angle_deg=ang,
                ell_area=float(np.pi*ax_a*ax_b),
                bbox_x0=x0, bbox_y0=y0, bbox_x1=x1, bbox_y1=y1,
                bbox_area=float((x1-x0)*(y1-y0)),
                nose_x=lm[NOSE_TIP,0], nose_y=lm[NOSE_TIP,1],
                eyeL_x=lm[EYE_L,0].mean(), eyeL_y=lm[EYE_L,1].mean(),
                eyeR_x=lm[EYE_R,0].mean(), eyeR_y=lm[EYE_R,1].mean(),
                chin_x=lm[CHIN,0], chin_y=lm[CHIN,1],
                n_poly=len(poly), img_w=im.shape[1], img_h=im.shape[0]))
            for k,(px,py) in enumerate(poly):
                polys.append(dict(stem=stem, face_id=fid, k=k,
                                  x_img=px, y_img=py,
                                  x_scr=px+OFFSET[0], y_scr=py+OFFSET[1]))
            for k,(px,py) in enumerate(lm):
                lms.append(dict(stem=stem, face_id=fid, pt=k,
                                x_img=px, y_img=py,
                                x_scr=px+OFFSET[0], y_scr=py+OFFSET[1]))
        if n % 10 == 0 or n == len(files):
            print(f"  {n}/{len(files)}", flush=True)

    B = pd.DataFrame(rows)
    if B.empty:
        print("no faces detected"); return
    lo, hi = np.percentile(B.bbox_area, [SIZE_LO, SIZE_HI])
    B["size_flag"] = np.where(B.bbox_area < lo, "small",
                      np.where(B.bbox_area > hi, "large", ""))
    B.to_csv(out/"face_bounds.csv", index=False)
    pd.DataFrame(polys).to_csv(out/"face_polygons.csv", index=False)
    pd.DataFrame(lms).to_csv(out/"face_landmarks.csv", index=False)
    print(f"\n{len(B)} faces over {B.stem.nunique()} images "
          f"({len(B)/B.stem.nunique():.1f} per image)")
    print(f"  bbox area  p10={np.percentile(B.bbox_area,10):.0f} "
          f"p50={np.percentile(B.bbox_area,50):.0f} "
          f"p90={np.percentile(B.bbox_area,90):.0f} px^2")
    print(f"  ellipse semi-axes  a {B.ell_a.median():.0f}  b {B.ell_b.median():.0f} px "
          f"({B.ell_a.median()/44:.1f} x {B.ell_b.median()/44:.1f} deg)")
    print(f"  flagged: {(B.size_flag!='').sum()} "
          f"({(B.size_flag=='small').sum()} small, {(B.size_flag=='large').sum()} large)")
    multi = B.groupby("stem").size()
    if (multi>1).any():
        print(f"  multiple faces: " + ", ".join(
            f"{s}={n}" for s,n in multi[multi>1].items()))

    if not a.no_overlay:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly, Ellipse
        from PIL import Image as PILImage
        P = pd.DataFrame(polys)
        for stem, g in B.groupby("stem"):
            f = [x for x in files if Path(x).stem == stem][0]
            im = PILImage.open(f).convert("RGB")
            fig, ax = plt.subplots(figsize=(im.size[0]/110, im.size[1]/110))
            ax.imshow(im)
            for r in g.itertuples():
                pg = P[(P.stem==stem)&(P.face_id==r.face_id)][["x_img","y_img"]].to_numpy()
                ax.add_patch(MplPoly(pg, fill=False, lw=1.4, ec="#00ff88"))
                ax.add_patch(Ellipse((r.cx_img, r.cy_img), 2*r.ell_a, 2*r.ell_b,
                                     angle=r.ell_angle_deg, fill=False,
                                     lw=1.6, ec="#ff3355"))
                ax.plot(r.cx_img, r.cy_img, "+", color="#ff3355", ms=10)
                ax.plot(r.nose_x, r.nose_y, ".", color="yellow", ms=5)
                ax.plot([r.eyeL_x, r.eyeR_x], [r.eyeL_y, r.eyeR_y], ".",
                        color="cyan", ms=5)
                if r.size_flag:
                    ax.annotate(r.size_flag, (r.cx_img, r.cy_img), color="orange",
                                fontsize=8, xytext=(6,-10),
                                textcoords="offset points")
            L = pd.DataFrame(lms)
            ll = L[L.stem==stem]
            ax.plot(ll.x_img, ll.y_img, ".", color="white", ms=1.5, alpha=.7)
            ax.set_axis_off()
            ax.set_title(f"{stem} — {len(g)} face(s); green polygon, red ellipse",
                         fontsize=9)
            fig.tight_layout()
            fig.savefig(out/"overlay"/f"{stem}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
        print(f"wrote {B.stem.nunique()} overlays to {out}/overlay")
    print(f"wrote {out}/face_bounds.csv, face_polygons.csv, face_landmarks.csv")


if __name__ == "__main__":
    main()
