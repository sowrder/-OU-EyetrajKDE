#!/usr/bin/env python3
"""
ou_visits.py — stage 3 of the segmentation pipeline: p̂, basins, membership
and visits, per participant × image.

ROLE IN THE DOCUMENT
  Implements §2 and the estimation of §3 for one participant-session and one
  bandwidth h.  reproduce.py runs it three times per session (§4.1):
  h = 26 px, h = ĥ_im per image, and h = ĥ_M.

SAMPLES
  Stimulus-phase samples of non-practice trials, cleaned by clean_mask():
  out-of-screen and missing samples removed; ±80 ms around every missing
  sample; each EyeLink saccade removed with a ±4 ms margin; samples above
  the velocity cap removed.  Trials with fewer than 20 retained samples are
  dropped, and images with fewer than 200 pooled samples.  One trial per
  image, so p̂ is per participant × image.

DIFFUSION
  σ² = mean of ‖S_{k+1} − S_k‖² / 2Δt over adjacent retained pairs with
  Δt ≤ 4 ms, and D = σ²/2.  §5 shows this adjacent-sample estimate is set
  largely by the tracker's filtering; everything below that the document
  interprets is D-invariant.

KDE AND BASINS  (§2)
  p̂ on the grid G (spacing h_g = 16 px, half-shifted), Gaussian kernel of
  width h.  log p̂ floored at 1e-7.  Derivatives by finite differences.
  A node is a basin centre μ_k if it passes the three conditions:
    1  p̂ ≥ 60th percentile of p̂ over G
    2  p̂ ≥ max over its 8 neighbours
    3  λ_min(Λ_k) > 0, Λ_k = −D [∇∇ᵀ log p̂]_{μ_k}, symmetrised
  θ_1, θ_2 are the eigenvalues of Λ_k and θ their mean; basin_px =
  σ / √(2θ).  λ = θ/D is nearly constant across basins (§5), so θ largely
  restates σ².

MEMBERSHIP  (Q_k)
  A sample x belongs to the basin minimising

      Q_k(x) = (x − μ_k)ᵀ Λ_k (x − μ_k) / σ²   =  ½ (x − μ_k)ᵀ [−∇∇ᵀ log p̂]_{μ_k} (x − μ_k)

  if that minimum is ≤ 1, otherwise it is unlabelled.  D cancels, so the
  partition depends on p̂ alone.

FACE ATTRIBUTE
  on_face: min_v Q_v(μ_k) ≤ 1 against the ellipses of facelabel.py,
  evaluated at the basin centre.  reproduce.py recomputes the identity and
  the depth q = min_v Q_v(μ_k) from the same ellipses.

VISITS  (§3)
  A visit is a maximal run of samples with one label within a trial, broken
  by a label change or a gap > 60 ms, kept if it lasts ≥ 8 ms.  The
  re-entry index r counts earlier visits to the same basin in the same image.

NOT INTERPRETED
  The drift f = D ∇ log p̂, the work W = Σ f · ΔX and |f| are written for
  completeness; the document does not interpret them (§5).

OUTPUT (--outdir)
  visits.csv         one row per visit: basin, μ_k, θ, θ_1, θ_2, eigenvectors,
                     anisotropy, basin_px, on_face, entry, exit, dwell, r,
                     σ², h
  visit_summary.csv  per participant × stimulus set
  duration_dist.png  quick-look figure
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
SCREEN = (1280, 1024)
GUARD_MS = 80.0
VEL_MAX_PX = 1000.0 * 44.0
GRID_PX = 16.0
KDE_PX = 26.0
MIN_DENS = 1e-7
MIN_VISIT_MS = 8.0
MAX_GAP_MS = 60.0


def load_pid(ext, pid, ses):
    f = list(ext.glob(f"{pid}_ses-{ses}_samples.*"))
    if not f: return None, None
    f = f[0]
    S = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
    sc = ext / f"{pid}_ses-{ses}_saccades.csv"
    return S, (pd.read_csv(sc) if sc.exists() else pd.DataFrame())


def clean_mask(t, x, y, sacc):
    """retained samples: on screen, finite, outside ±GUARD_MS of any missing
    sample, outside every EyeLink saccade ±4 ms, below the velocity cap"""
    ok = np.isfinite(x) & np.isfinite(y)
    ok &= (x > 0) & (x < SCREEN[0]) & (y > 0) & (y < SCREEN[1])
    bad = ~np.isfinite(x)
    if bad.any():
        for i in np.where(bad)[0]:
            ok &= ~((t >= t[i]-GUARD_MS) & (t <= t[i]+GUARD_MS))
    if len(sacc):
        lo, hi = t.min()-GUARD_MS, t.max()+GUARD_MS
        s = sacc[(sacc.end_ms >= lo) & (sacc.start_ms <= hi)]
        for s0, s1 in zip(s.start_ms.values, s.end_ms.values):
            ok &= ~((t >= s0-4) & (t <= s1+4))
    v = np.zeros(len(t)); dt = np.diff(t)/1000.0
    v[1:] = np.hypot(np.diff(x), np.diff(y))/np.where(dt > 0, dt, np.inf)
    return ok & (v <= VEL_MAX_PX)


def silverman_h(X):
    """ĥ_im = T_im^(-1/6) σ̂_im with σ̂ = √(σ̂_x σ̂_y), §4.1.  At d = 2 the
    constant 4/(d+2) is 1."""
    m = len(X)
    sd = X.std(axis=0, ddof=1)
    sg = float(np.sqrt(max(sd[0]*sd[1], 1e-12)))
    return sg*(4.0/((2+2)*m))**(1.0/(2+4))


def field(X, D_coef, h):
    """p̂ on G with kernel width h, log p̂ (floored), and the drift
    f = D ∇ log p̂ (written, not interpreted)"""
    gx = np.arange(GRID_PX/2, SCREEN[0], GRID_PX)
    gy = np.arange(GRID_PX/2, SCREEN[1], GRID_PX)
    GX, GY = np.meshgrid(gx, gy)
    P = np.zeros(GX.shape)
    inv = -0.5/h**2; norm = 1.0/(2*np.pi*h**2*len(X))
    for j in range(GX.shape[0]):
        d2 = ((X[:,0][None,:]-GX[j][:,None])**2 +
              (X[:,1][None,:]-GY[j][:,None])**2)
        P[j] = np.exp(inv*d2).sum(1)*norm
    L = np.log(np.maximum(P, MIN_DENS))
    dLy, dLx = np.gradient(L, gy, gx)
    return gx, gy, P, L, D_coef*dLx, D_coef*dLy


def sample_grid(gx, gy, Z, px, py):
    """bilinear interpolation of a grid field at the samples"""
    fx = np.clip((px-gx[0])/(gx[1]-gx[0]), 0, len(gx)-1.001)
    fy = np.clip((py-gy[0])/(gy[1]-gy[0]), 0, len(gy)-1.001)
    i0, j0 = fx.astype(int), fy.astype(int)
    a, b = fx-i0, fy-j0
    return ((1-a)*(1-b)*Z[j0,i0] + a*(1-b)*Z[j0,i0+1] +
            (1-a)*b*Z[j0+1,i0] + a*b*Z[j0+1,i0+1])


def maxima(gx, gy, P, L, D_coef, sigma2):
    """the three basin conditions of §2; returns μ_k, θ, θ_1, θ_2, the
    eigenvectors of Λ_k, p̂(μ_k) and basin_px = σ/√(2θ)"""
    Lyy, Lyx = np.gradient(np.gradient(L, gy, axis=0), gy, gx)
    Lxy, Lxx = np.gradient(np.gradient(L, gx, axis=1), gy, gx)
    out = []; ny, nx = P.shape; thr = np.nanpercentile(P, 60)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            if P[j,i] < thr or P[j,i] < P[j-1:j+2, i-1:i+2].max(): continue
            H = -np.array([[Lxx[j,i], Lxy[j,i]], [Lyx[j,i], Lyy[j,i]]])
            H = 0.5*(H+H.T)
            w, V = np.linalg.eigh(H)
            if w.min() <= 0: continue
            th = D_coef*w
            out.append(dict(mu_x=float(gx[i]), mu_y=float(gy[j]),
                            theta=float(th.mean()),
                            th1=float(th[0]), th2=float(th[1]),
                            v11=float(V[0,0]), v21=float(V[1,0]),
                            v12=float(V[0,1]), v22=float(V[1,1]),
                            dens=float(P[j,i]),
                            basin_px=float(np.sqrt(sigma2/(2*th.mean())))))
    return pd.DataFrame(out)


def assign(X, M, sigma2, mode):
    """label = argmin_k Q_k(x); 'inside' leaves x unlabelled if min Q_k > 1"""
    if M.empty: return np.full(len(X), -1)
    Q = np.empty((len(X), len(M)))
    for k, r in enumerate(M.itertuples()):
        d = X - np.array([r.mu_x, r.mu_y])
        V = np.array([[r.v11, r.v12], [r.v21, r.v22]])
        Th = V @ np.diag([r.th1, r.th2]) @ V.T
        Q[:, k] = np.einsum("ni,ij,nj->n", d, Th, d)/sigma2
    lab = Q.argmin(1)
    if mode == "inside":
        lab = np.where(Q[np.arange(len(X)), lab] <= 1.0, lab, -1)
    return lab


def in_ellipse(px, py, cx, cy, a, b, ang):
    """Q_v at (px, py) for one face ellipse, in its rotated frame"""
    t = np.radians(ang); ct, st = np.cos(t), np.sin(t)
    dx, dy = px-cx, py-cy
    u = dx*ct + dy*st; v = -dx*st + dy*ct
    return (u/a)**2 + (v/b)**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nas", default="/mnt/n/Soma/OU_EpochDistribution_Pipeline")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--ses", default="05")
    ap.add_argument("--faces", action="append", default=[])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--membership", choices=["inside","nearest"], default="inside")
    ap.add_argument("--images", default="")
    ap.add_argument("--ext", default=None,
                    help="dir holding <pid>_ses-<ss>_samples.*")
    ap.add_argument("--root", default=".")
    ap.add_argument("--kde-px", type=float, default=KDE_PX,
                    help="bandwidth h in px (fixed mode)")
    ap.add_argument("--kde-mode", choices=["fixed","silverman"], default="fixed",
                    help="fixed: use --kde-px.  silverman: per-image rule")
    a = ap.parse_args()

    root = Path(a.root)
    ext = Path(a.ext) if a.ext else root/"output"/"extracted"/f"ses-{a.ses}"
    out = Path(a.outdir) if a.outdir else \
        root/"output"/"ou_visits"/f"{a.pid}_ses-{a.ses}"
    out.mkdir(parents=True, exist_ok=True)

    FB = []
    for d in a.faces:
        f = Path(d)/"face_bounds.csv"
        if f.exists(): FB.append(pd.read_csv(f, dtype={"stem": str}))
    FB = pd.concat(FB, ignore_index=True) if FB else pd.DataFrame()

    S, C = load_pid(ext, a.pid, a.ses)
    if S is None: raise FileNotFoundError(f"no samples for {a.pid} ses-{a.ses}")
    ST = S[(S.phase=="stimulus") & (~S.practice.astype(bool))]
    want = [w.strip() for w in a.images.split(",") if w.strip()]
    print(f"{a.pid}/ses-{a.ses}: {len(ST)} samples, {ST.image.nunique()} images",
          flush=True)

    rows = []
    for img, gi in ST.groupby("image"):
        stem = str(img).rsplit(".",1)[0]
        if want and stem not in want: continue
        keep = []
        for tr, gt in gi.groupby("trial_i"):
            gt = gt.sort_values("t_ms")
            t = gt.t_ms.to_numpy().astype(float)
            x = gt.x.to_numpy().astype(float); y = gt.y.to_numpy().astype(float)
            m = clean_mask(t, x, y, C)
            if m.sum() < 20: continue
            keep.append((tr, t[m], x[m], y[m]))
        if not keep: continue
        X = np.column_stack([np.concatenate([k[2] for k in keep]),
                             np.concatenate([k[3] for k in keep])])
        if len(X) < 200: continue
        dd = []
        for _, t, x, y in keep:
            adj = np.diff(t) <= 4
            if adj.any():
                dt = np.diff(t)[adj]/1000.0
                dd.append((np.diff(x)[adj]**2 + np.diff(y)[adj]**2)/(2*dt))
        sigma2 = float(np.mean(np.concatenate(dd))) if dd else 1e4
        D_coef = sigma2/2.0
        h_use = silverman_h(X) if a.kde_mode == "silverman" else a.kde_px
        gx, gy, P, L, Fx, Fy = field(X, D_coef, h_use)
        M = maxima(gx, gy, P, L, D_coef, sigma2)
        if M.empty:
            print(f"  [{stem:24}] no basins"); continue
        faces = FB[FB.stem == stem] if len(FB) else pd.DataFrame()
        onf = []
        for r in M.itertuples():
            hit = False
            for fr in faces.itertuples():
                hit |= in_ellipse(r.mu_x, r.mu_y, fr.cx_scr, fr.cy_scr,
                                  fr.ell_a, fr.ell_b, fr.ell_angle_deg) <= 1.0
            onf.append(hit)
        M["on_face"] = onf

        seen = {}; nv0 = len(rows)
        for tr, t, x, y in keep:
            XY = np.column_stack([x, y])
            lab = assign(XY, M, sigma2, a.membership)
            fx = sample_grid(gx, gy, Fx, x, y)
            fy = sample_grid(gx, gy, Fy, x, y)
            # a visit ends on a label change or a gap > MAX_GAP_MS (§3)
            brk = (np.diff(lab) != 0) | (np.diff(t) > MAX_GAP_MS)
            cut = np.r_[0, np.where(brk)[0]+1, len(lab)]
            for i0, i1 in zip(cut[:-1], cut[1:]):
                k = lab[i0]
                if k < 0: continue
                dur = t[i1-1] - t[i0]
                if dur < MIN_VISIT_MS: continue
                sl = slice(i0, i1)
                dx = np.diff(x[sl]); dy = np.diff(y[sl])
                W = float(np.sum(fx[sl][:-1]*dx + fy[sl][:-1]*dy))
                plen = float(np.sum(np.hypot(dx, dy)))
                r = M.iloc[k]
                seen[k] = seen.get(k, -1) + 1
                rows.append(dict(
                    pid=a.pid, ses=f"ses-{a.ses}", image=img, stem=stem,
                    trial_i=int(tr), basin=int(k), re=seen[k],
                    entry_ms=float(t[i0]), exit_ms=float(t[i1-1]),
                    dwell_ms=float(dur), n_samp=int(i1-i0),
                    mu_x=float(r.mu_x), mu_y=float(r.mu_y),
                    theta=float(r.theta), basin_px=float(r.basin_px),
                    on_face=bool(r.on_face), dens=float(r.dens),
                    work=W, work_per_ms=W/max(dur, 1.0),
                    f_mean=float(np.hypot(fx[sl], fy[sl]).mean()),
                    path_px=plen, sigma2=sigma2, h_px=float(h_use),
                    n_basins=int(len(M)),
                    # the full Λ_k: eigenvalues θ_1, θ_2 and eigenvectors
                    th1=float(r.th1), th2=float(r.th2),
                    v11=float(r.v11), v21=float(r.v21),
                    v12=float(r.v12), v22=float(r.v22),
                    aniso=float(max(r.th1,r.th2)/max(min(r.th1,r.th2),1e-12)),
                    ang_deg=float(np.degrees(np.arctan2(r.v21, r.v11)))))
        print(f"  [{stem:24}] h={h_use:5.1f}  {len(M):2d} basins  "
              f"{len(rows)-nv0:4d} visits",
              flush=True)

    if not rows:
        print("no visits"); return
    V = pd.DataFrame(rows)
    V["kind"] = np.where(V.stem.str.fullmatch(r"\d{4}"), "IAPS", "ABM")
    V.to_csv(out/"visits.csv", index=False)

    print(f"\n{len(V)} visits over {V.stem.nunique()} images")
    print(f"  dwell ms   p10={V.dwell_ms.quantile(.1):6.0f} "
          f"p50={V.dwell_ms.median():6.0f} p90={V.dwell_ms.quantile(.9):6.0f}")
    print(f"  theta 1/s  p10={V.theta.quantile(.1):6.2f} "
          f"p50={V.theta.median():6.2f} p90={V.theta.quantile(.9):6.2f}")
    print(f"  |work|     p50={V.work.abs().median():8.0f} "
          f"p90={V.work.abs().quantile(.9):8.0f}")
    print(f"  on-face    {100*V.on_face.mean():.1f}% of visits, "
          f"{100*V[V.on_face].dwell_ms.sum()/max(V.dwell_ms.sum(),1):.1f}% of dwell")
    print("\nby re-entry index:")
    print(V.groupby(V.re.clip(0,8)).agg(
        n=("dwell_ms","size"), dwell=("dwell_ms","median"),
        theta=("theta","median"), work=("work","median"),
        on_face=("on_face","mean")).to_string(float_format=lambda v:f"{v:.3f}"))
    print("\nby stimulus type:")
    print(V.groupby("kind").agg(
        n=("dwell_ms","size"), dwell=("dwell_ms","median"),
        theta=("theta","median"), work=("work","median"),
        on_face=("on_face","mean")).to_string(float_format=lambda v:f"{v:.3f}"))
    print("\non-face vs off-face:")
    print(V.groupby("on_face").agg(
        n=("dwell_ms","size"), dwell=("dwell_ms","median"),
        theta=("theta","median"), work=("work","median"),
        basin_deg=("basin_px", lambda s: s.median()/44)
        ).to_string(float_format=lambda v:f"{v:.3f}"))
    V.groupby(["pid","ses","kind"]).agg(
        n_visits=("dwell_ms","size"), dwell_med=("dwell_ms","median"),
        theta_med=("theta","median"), work_med=("work","median"),
        on_face=("on_face","mean")).reset_index().to_csv(
        out/"visit_summary.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for kd, g in V.groupby("kind"):
        ax[0].hist(np.log10(g.dwell_ms.clip(1)), bins=60, histtype="step",
                   density=True, label=kd)
    ax[0].set_xlabel("log10 dwell (ms)"); ax[0].set_ylabel("density")
    ax[0].set_title("visit duration"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for lab, g in V.groupby("on_face"):
        r = g.groupby(g.re.clip(0,8)).dwell_ms.median()
        ax[1].plot(r.index, r.values, marker="o",
                   label="on face" if lab else "off face")
    ax[1].set_xlabel("re-entry index"); ax[1].set_ylabel("median dwell (ms)")
    ax[1].set_title("dwell vs re-entry"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    for lab, g in V.groupby("on_face"):
        ax[2].hist(np.log10(g.work.abs().clip(1)), bins=60, histtype="step",
                   density=True, label="on face" if lab else "off face")
    ax[2].set_xlabel("log10 |OU work|"); ax[2].set_title("work distribution")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    ax[3].scatter(V.dwell_ms, V.work.abs(), s=2, alpha=.12,
                  c=np.where(V.on_face, "tab:red", "tab:blue"))
    ax[3].set_xscale("log"); ax[3].set_yscale("log")
    ax[3].set_xlabel("dwell (ms)"); ax[3].set_ylabel("|OU work|")
    ax[3].set_title("work vs dwell (red = on face)"); ax[3].grid(alpha=.3)
    fig.suptitle(f"{a.pid} ses-{a.ses} — basin visits, membership={a.membership}")
    fig.tight_layout(); fig.savefig(out/"duration_dist.png", dpi=130,
                                    bbox_inches="tight")
    print(f"\nwrote {out}/visits.csv, visit_summary.csv, duration_dist.png")


if __name__ == "__main__":
    main()
