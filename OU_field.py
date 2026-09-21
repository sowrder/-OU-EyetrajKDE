#!/usr/bin/env python3
"""
OU_field.py — per-image field figures.  Diagnostic only: no number in the
document comes from this script, and reproduce.py does not run it.

WHAT IT DRAWS
  For each image: the stimulus, p̂ as filled contours, the gaze path coloured
  by a local pull rate, the face ellipses Q_v, and the basin centres μ_k
  (green when min_v Q_v(μ_k) ≤ 1).  Uses the same KDE and basin conditions
  as ou_visits.py (§2), with h passed as --kde-px.

DIFFERENCES FROM ou_visits.py, WHICH IS WHAT THE DOCUMENT USES
  σ²          from increments between samples adjacent in the retained
              index (np.diff(i) == 1) rather than Δt ≤ 4 ms, so values can
              differ slightly from visits.csv.
  path colour a local rate D · (−½ ∇² log p̂), the mean of the Hessian's
              diagonal, defined only where it is positive.  It is not Λ_k
              and not the θ of the document.
  on-face     share of gaze samples inside any Q_v ≤ 1, per image; the
              document tests basin centres instead (§4.2).

OUTPUT (--outdir)
  ou_field_sinks.csv    per image × basin: μ_k, θ, basin size, on_face,
                        distance to the nearest face in ellipse radii
  ou_field_images.csv   per image: samples, faces, basins, gaze on-face share
  field_<stem>.png      the figure
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
OFFSET = np.array([128.0, 128.0])
SCREEN = (1280, 1024)
GUARD_MS = 80.0
VEL_MAX = 1000.0
GRID_PX = 16.0
KDE_PX = 26.0          # density kernel sd
MIN_DENS = 1e-7


def load_pid(ext: Path, pid: str, ses: str):
    f = list(ext.glob(f"{pid}_ses-{ses}_samples.*"))
    if not f:
        raise FileNotFoundError(f"no samples for {pid} ses-{ses} in {ext}")
    f = f[0]
    S = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
    sc = ext / f"{pid}_ses-{ses}_saccades.csv"
    C = pd.read_csv(sc) if sc.exists() else pd.DataFrame()
    return S, C


def clean_trial(g, sacc):
    t = g.t_ms.to_numpy().astype(float)
    x = g.x.to_numpy().astype(float); y = g.y.to_numpy().astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    ok &= (x > 0) & (x < SCREEN[0]) & (y > 0) & (y < SCREEN[1])
    if len(sacc):
        lo, hi = t.min() - GUARD_MS, t.max() + GUARD_MS
        s = sacc[(sacc.end_ms >= lo) & (sacc.start_ms <= hi)]
        for s0, s1 in zip(s.start_ms.values, s.end_ms.values):
            ok &= ~((t >= s0 - 4) & (t <= s1 + 4))
    # ±GUARD_MS around every missing sample, as in ou_visits.clean_mask
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.where(bad)[0]
        for i in idx:
            ok &= ~((t >= t[i] - GUARD_MS) & (t <= t[i] + GUARD_MS))
    v = np.zeros(len(t)); dtv = np.diff(t) / 1000.0
    v[1:] = np.hypot(np.diff(x), np.diff(y)) / np.where(dtv > 0, dtv, np.inf)
    ok &= v <= VEL_MAX * 44.0
    i = np.where(ok)[0]
    if len(i) < 3: return None, None
    adj = np.diff(i) == 1
    i0, i1 = i[:-1][adj], i[1:][adj]
    if len(i0) < 3: return np.column_stack([x[i], y[i]]), None
    dt = (t[i1] - t[i0]) / 1000.0
    ok2 = dt > 0
    dX = np.column_stack([x[i1]-x[i0], y[i1]-y[i0]])[ok2]
    return np.column_stack([x[i], y[i]]), (dX, dt[ok2])


def density_field(X, D_coef, h=KDE_PX):
    """p̂ on G, log p̂, the drift f = D ∇ log p̂, and a local pull rate for
    drawing only"""
    gx = np.arange(GRID_PX/2, SCREEN[0], GRID_PX)
    gy = np.arange(GRID_PX/2, SCREEN[1], GRID_PX)
    GX, GY = np.meshgrid(gx, gy)
    P = np.zeros(GX.shape)
    inv = -0.5 / h**2
    norm = 1.0 / (2*np.pi*h**2*len(X))
    for j in range(GX.shape[0]):
        d2 = ((X[:,0][None,:] - GX[j][:,None])**2 +
              (X[:,1][None,:] - GY[j][:,None])**2)
        P[j] = np.exp(inv*d2).sum(1) * norm
    L = np.log(np.maximum(P, MIN_DENS))
    dLy, dLx = np.gradient(L, gy, gx)
    F = np.dstack([D_coef*dLx, D_coef*dLy])       # px/s
    # local rate for colouring the path: D · (−½ ∇² log p̂), kept where
    # positive.  Not Λ_k and not the θ of the document.
    Lyy, _ = np.gradient(np.gradient(L, gy, axis=0), gy, gx)
    _, Lxx = np.gradient(np.gradient(L, gx, axis=1), gy, gx)
    TH = D_coef * (-(Lxx + Lyy) / 2.0)
    TH = np.where(TH > 0, TH, np.nan)
    return gx, gy, GX, GY, P, L, F, TH


def find_maxima(gx, gy, P, L, D_coef, sigma2):
    Lyy, Lyx = np.gradient(np.gradient(L, gy, axis=0), gy, gx)
    Lxy, Lxx = np.gradient(np.gradient(L, gx, axis=1), gy, gx)
    out = []; ny, nx = P.shape
    thr = np.nanpercentile(P, 60)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            if P[j,i] < thr: continue
            if P[j,i] < P[j-1:j+2, i-1:i+2].max(): continue
            H = -np.array([[Lxx[j,i], Lxy[j,i]], [Lyx[j,i], Lyy[j,i]]])
            H = 0.5*(H + H.T)
            ev = np.linalg.eigvalsh(H)
            if ev.min() <= 0: continue                # not a maximum
            th = D_coef*ev                            # pull rates, 1/s
            basin = np.sqrt(sigma2/(2*np.maximum(th, EPS)))
            out.append(dict(mu_x=float(gx[i]), mu_y=float(gy[j]),
                            dens=float(P[j,i]),
                            theta=float(th.mean()),
                            theta_min=float(th.min()), theta_max=float(th.max()),
                            anisotropy=float(th.max()/max(th.min(), EPS)),
                            basin_px=float(basin.mean()),
                            basin_min_px=float(basin.min()),
                            basin_max_px=float(basin.max())))
    return pd.DataFrame(out)


def in_ellipse(px, py, cx, cy, a, b, ang):
    t = np.radians(ang); ct, st = np.cos(t), np.sin(t)
    dx, dy = px-cx, py-cy
    u = dx*ct + dy*st; v = -dx*st + dy*ct
    return (u/a)**2 + (v/b)**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nas", default="/mnt/n/Soma/OU_EpochDistribution_Pipeline")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--ses", default="05")
    ap.add_argument("--faces", action="append", required=True,
                    metavar="DIR", help="face_bounds dir; repeatable "
                    "(IAPS and the participant's ABM)")
    ap.add_argument("--imgdir", action="append", required=True,
                    metavar="DIR", help="image dir; repeatable, same order")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--images", default="")
    ap.add_argument("--max-figs", type=int, default=40)
    ap.add_argument("--ext", default=None)
    ap.add_argument("--root", default=".")
    ap.add_argument("--kde-px", type=float, default=KDE_PX)
    a = ap.parse_args()

    root = Path(a.root)
    ext = Path(a.ext) if a.ext else root/"output"/"extracted"/f"ses-{a.ses}"
    out = Path(a.outdir) if a.outdir else \
        root/"output"/"ou_field"/f"{a.pid}_ses-{a.ses}"
    out.mkdir(parents=True, exist_ok=True)

    FB = []
    for d in a.faces:
        f = Path(d)/"face_bounds.csv"
        if f.exists(): FB.append(pd.read_csv(f, dtype={"stem": str}))
        else: print(f"  no face_bounds.csv in {d}")
    FB = pd.concat(FB, ignore_index=True) if FB else pd.DataFrame()
    IMG = {}
    for d in a.imgdir:
        for f in glob.glob(f"{d}/*"):
            if Path(f).suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                IMG.setdefault(Path(f).stem, f)
    print(f"{len(FB)} face rows, {len(IMG)} images on disk")

    S, C = load_pid(ext, a.pid, a.ses)
    ST = S[(S.phase=="stimulus") & (~S.practice.astype(bool))]
    print(f"{a.pid}/ses-{a.ses}: {len(ST)} stimulus samples, "
          f"{ST.image.nunique()} images")

    want = [w.strip() for w in a.images.split(",") if w.strip()]
    sinks, imgs, figq = [], [], []
    for img, gi in ST.groupby("image"):
        stem = str(img).rsplit(".",1)[0]
        if want and stem not in want and str(img) not in want: continue
        XS, DS, DT = [], [], []
        for tr, gt in gi.groupby("trial_i"):
            X, inc = clean_trial(gt.sort_values("t_ms"), C)
            if X is None: continue
            XS.append(X)
            if inc is not None: DS.append(inc[0]); DT.append(inc[1])
        if not XS: continue
        X = np.vstack(XS)
        if len(X) < 200:
            print(f"  [{stem}] {len(X)} samples — thin"); continue
        # σ² from index-adjacent retained increments; ou_visits.py uses
        # Δt ≤ 4 ms instead, which is what the document reports
        if DS:
            dX = np.vstack(DS); dt = np.concatenate(DT)
            sigma2 = float(np.mean((dX**2).sum(1)/(2*dt)))    # px^2/s
        else:
            sigma2 = 1e4
        D_coef = sigma2/2.0
        gx, gy, GX, GY, P, L, F, TH = density_field(X, D_coef, a.kde_px)
        M = find_maxima(gx, gy, P, L, D_coef, sigma2)

        faces = FB[FB.stem == stem]
        onf = np.zeros(len(X), bool)
        for r in faces.itertuples():
            onf |= in_ellipse(X[:,0], X[:,1], r.cx_scr, r.cy_scr,
                              r.ell_a, r.ell_b, r.ell_angle_deg) <= 1.0
        of = float(onf.mean())

        if len(M):
            M = M.sort_values("dens", ascending=False).reset_index(drop=True)
            onface, dmin = [], []
            for r in M.itertuples():
                best, hit = np.inf, False
                for fr in faces.itertuples():
                    q = in_ellipse(r.mu_x, r.mu_y, fr.cx_scr, fr.cy_scr,
                                   fr.ell_a, fr.ell_b, fr.ell_angle_deg)
                    best = min(best, float(np.sqrt(q)))
                    hit |= q <= 1.0
                onface.append(hit); dmin.append(best if np.isfinite(best) else np.nan)
            M["on_face"] = onface; M["d_face_radii"] = dmin
            M.insert(0,"stem",stem); M.insert(0,"pid",a.pid)
            M.insert(2,"ses",f"ses-{a.ses}")
            sinks.append(M)
        n_on = int(M.on_face.sum()) if len(M) else 0
        imgs.append(dict(pid=a.pid, ses=f"ses-{a.ses}", stem=stem,
                         n_samples=len(X), n_faces=len(faces),
                         n_sinks=len(M), n_sinks_on_face=n_on,
                         on_face_frac=of, sigma2=sigma2))
        print(f"  [{stem:26}] n={len(X):6d} faces={len(faces):2d} "
              f"sinks={len(M):2d} on-face={n_on:2d} "
              f"gaze on-face={100*of:5.1f}%  sig2={sigma2:.0f}", flush=True)
        if len(figq) < a.max_figs:
            figq.append((stem, X, gx, gy, GX, GY, P, F, TH, M, faces, sigma2))

    if sinks:
        A = pd.concat(sinks, ignore_index=True)
        A.to_csv(out/"ou_field_sinks.csv", index=False)
    if imgs:
        I = pd.DataFrame(imgs); I.to_csv(out/"ou_field_images.csv", index=False)
        print(f"\n{I.n_sinks.sum()} sinks over {len(I)} images")
        print(f"  gaze on-face: median {100*I.on_face_frac.median():.1f}%  "
              f"[{100*I.on_face_frac.min():.1f}-{100*I.on_face_frac.max():.1f}]")
        w = I[I.n_sinks > 0]
        if len(w):
            print(f"  sinks on a face: {w.n_sinks_on_face.sum()}/{w.n_sinks.sum()} "
                  f"({100*w.n_sinks_on_face.sum()/max(w.n_sinks.sum(),1):.0f}%)")
        if sinks:
            A = pd.concat(sinks, ignore_index=True)
            for q in (10,50,90):
                print(f"  p{q}: theta={A.theta.quantile(q/100):7.2f}/s  "
                      f"basin={A.basin_px.quantile(q/100)/44:5.2f}deg  "
                      f"d_face={A.d_face_radii.quantile(q/100):5.2f} radii")

    if figq:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Polygon as MplPoly
        from matplotlib.collections import LineCollection
        from PIL import Image as PILImage

        def at(gx, gy, Z, px, py):
            """bilinear sample of a grid field at scattered points."""
            fx = np.clip((px - gx[0]) / (gx[1]-gx[0]), 0, len(gx)-1.001)
            fy = np.clip((py - gy[0]) / (gy[1]-gy[0]), 0, len(gy)-1.001)
            i0, j0 = fx.astype(int), fy.astype(int)
            a, b = fx - i0, fy - j0
            return ((1-a)*(1-b)*Z[j0, i0] + a*(1-b)*Z[j0, i0+1] +
                    (1-a)*b*Z[j0+1, i0] + a*b*Z[j0+1, i0+1])

        for stem, X, gx, gy, GX, GY, P, F, TH, M, faces, sigma2 in figq:
            fig, ax = plt.subplots(figsize=(10, 8))
            src = IMG.get(stem)
            if src:
                im = PILImage.open(src).convert("RGB")
                ax.imshow(im, extent=[OFFSET[0], OFFSET[0]+im.size[0],
                                      OFFSET[1]+im.size[1], OFFSET[1]],
                          alpha=.80)
            ax.contourf(GX, GY, np.log10(P+1e-9), levels=14,
                        cmap="magma", alpha=.32)

            # smooth the path before drawing: at 500 Hz consecutive samples
            # are sub-pixel apart and the line reads as noise
            k = 25
            if len(X) > 3*k:
                w = np.ones(k)/k
                px = np.convolve(X[:,0], w, "valid")
                py = np.convolve(X[:,1], w, "valid")
            else:
                px, py = X[:,0], X[:,1]

            # colour by pull strength along the path
            th = at(gx, gy, np.nan_to_num(TH, nan=0.0), px, py)
            seg = np.stack([np.column_stack([px[:-1], py[:-1]]),
                            np.column_stack([px[1:],  py[1:]])], axis=1)
            lc = LineCollection(seg, cmap="viridis", linewidths=2.0,
                                norm=plt.Normalize(0, np.nanpercentile(th, 95)+1e-9))
            lc.set_array(th[:-1]); ax.add_collection(lc)
            cb = fig.colorbar(lc, ax=ax, fraction=.03, pad=.01)
            cb.set_label("pull rate theta along path (1/s)", fontsize=8)
            cb.ax.tick_params(labelsize=7)

            # direction of travel, sparsely, so the arrows read as motion
            st = max(len(px)//45, 1)
            ax.quiver(px[:-1:st], py[:-1:st],
                      np.diff(px)[::st], -np.diff(py)[::st],
                      color="w", alpha=.85, width=.0022,
                      scale_units="xy", angles="xy", scale=.5)

            for fr in faces.itertuples():
                ax.add_patch(Ellipse((fr.cx_scr, fr.cy_scr),
                                     2*fr.ell_a, 2*fr.ell_b,
                                     angle=fr.ell_angle_deg, fill=False,
                                     lw=1.8, ec="#ff3355"))
            for r in M.itertuples():
                c = "#00ff66" if r.on_face else "yellow"
                ax.plot(r.mu_x, r.mu_y, "+", color=c, ms=13, mew=2)
                ax.annotate(f"θ{r.theta:.1f}", (r.mu_x, r.mu_y), color=c,
                            fontsize=7, xytext=(6,4), textcoords="offset points")
            ax.set_xlim(0, SCREEN[0]); ax.set_ylim(SCREEN[1], 0)
            ax.set_aspect("equal"); ax.set_axis_off()
            ax.set_title(f"{a.pid} ses-{a.ses}  {stem} — path coloured by pull, "
                         f"band = 1.96 sigma/sqrt(2theta); red = face, "
                         f"green + = sink on face", fontsize=8)
            fig.tight_layout()
            fig.savefig(out/f"field_{stem}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"wrote {len(figq)} figures to {out}")


if __name__ == "__main__":
    main()
