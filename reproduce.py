#!/usr/bin/env python3
"""
reproduce.py — regenerate every number in "Ornstein–Uhlenbeck Basin
Segmentation of Gaze Trajectories" from the raw data.

INPUTS (the only data the document depends on)
  eyedata/sub-<pid>/ses-<ss>/eye/*.asc     EyeLink recordings, one per session
  ImageTrials/IAPS/*.jpg                   the shared stimulus set
  ImageTrials/ABMimages/<pid>_trials4experiment/*
                                           participant-specific stimuli
  trajextract.py  facelabel.py  ou_visits.py  (OU_field.py, diagnostic only)

  Participant and session are taken from the directory, never the filename;
  the .asc names are inconsistent (see trajextract.py).

STAGES
  provenance   hashes of every input and script, library versions, the fixed
               parameters of §2 and §3, and the random seeds (§0 below)
  extract      trajextract.py per recording                         (§3)
  faces        facelabel.py per stimulus set                        (§3)
  samples      one pass over the cleaned samples: the per-image Silverman
               bandwidth ĥ_im and its mean ĥ_M (§4.1), and the shape
               statistics used in §5 (robust Silverman, increment kurtosis)
  runs         ou_visits.py for h = 26, ĥ_im and ĥ_M                (§4.1)
  prep         face identity, depth q and condition attached to every visit
  validate     every test in §4 and §5, written to output/reproduce/

  figures      one figure per section, from what validate wrote

  The first six stages are the segmentation pipeline; validate and figures
  test it.  --mode segment builds the partition only, --mode validate tests
  whatever exists without rebuilding it, --mode all (default) does both.
  Each stage skips work whose output exists unless --force; --stages gives
  an explicit list and overrides --mode.

DOCUMENT NOTATION USED BELOW
  S_k        a retained gaze sample
  p̂          the KDE on the grid G, bandwidth h
  μ_k, Λ_k   basin centre and pull matrix, Λ_k = -D[∇∇ᵀ log p̂]_{μ_k}
  θ          mean eigenvalue of Λ_k
  Q_k        basin membership ellipse, (x-μ_k)ᵀ Λ_k (x-μ_k) / σ² ≤ 1
  Q_v        face ellipse of face v, from the landmark fit
  O_i        basins in image i with min_v Q_v(μ_k) ≤ 1
  U_i        the same count for N uniform draws on G ∩ Ω_st
  A_iU       the same count for the participant's basins from another image
  A_iC       the same count for other participants' basins on the same image
  r          re-entry index; q = min_v Q_v(μ_k) the depth
"""
from __future__ import annotations
import argparse, glob, hashlib, importlib.util, json, os, platform, subprocess
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================== §0 constants
# Fixed parameters.  The stage scripts carry their own copies; these are
# recorded here so the provenance file states them, and checked against the
# imported ou_visits module so a silent edit to one side is caught.
PARAMS = dict(
    screen_px=(1280, 1024),         # Ω_T = [0,L_x]×[0,L_y]
    stim_px=(1024, 768),            # centred on screen, offset (128,128)
    stim_offset_px=(128, 128),
    grid_px=16.0,                   # h_g, grid spacing of G
    kde_px=26.0,                    # h, the bandwidth the results use
    density_floor_pct=60,           # basin condition 1
    min_dens=1e-7,                  # floor on p̂ before the log
    guard_ms=80.0,                  # excluded around every missing sample
    sacc_margin_ms=4.0,             # excluded around every EyeLink saccade
    vel_max_px_s=1000.0*44.0,       # velocity cap
    min_samples_trial=20,
    min_samples_image=200,
    sigma2_adj_ms=4.0,              # adjacent pairs for σ²
    min_visit_ms=8.0,               # §3 visit rule
    max_gap_ms=60.0,                # §3 visit rule
    truncation_ms=20.0,             # §3 truncated visit
    brief_ms=30.0,                  # §5 brief graze threshold
    floor_px=28.0,                  # basins below this sit at the smoothing floor
    match_px=26.0,                  # one kernel width, §4.3 stability
    ari_grid_ms=10.0,               # §4.3 partition agreement
    n_uniform=100,                  # N in U_i, §4.2
    n_alt=100,                      # draws of i' in A_iU
    n_perm=300,                     # permutation nulls, §4.3
    n_boot=2000,                    # bootstrap CI on O/A_U
    face_ellipse_scale=1.06,        # facelabel.py
)
SEEDS = dict(uniform=11, alt=12, jperm=13, facediff=14, boot=15)
CONDS = {"h26": ["--kde-mode", "fixed", "--kde-px", "26"],
         "hsilv": ["--kde-mode", "silverman"],
         "hmean": None}             # filled with ĥ_M after the samples stage
KEY = ["pid", "ses", "stem"]


# ================================================================== helpers
def say(*a):
    print(*a, flush=True)


def sh(cmd, label):
    say("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        say(f"  !! {label} exited {r.returncode}")
    return r.returncode == 0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_asc(root):
    out = []
    for f in sorted(glob.glob(str(root/"eyedata"/"sub-*"/"ses-*"/"eye"/"*.asc"))):
        p = Path(f).resolve().parts
        out.append((p[-4].replace("sub-", ""), p[-3].replace("ses-", ""), f))
    return out


def face_q(px, py, F):
    """Q_v evaluated at every point against every face in F (rows = faces),
    in each face's own rotated frame; returns the min over v and its argmin.
    Identical algebra to ou_visits.in_ellipse."""
    t = np.radians(F.ell_angle_deg.to_numpy())[None, :]
    dx = np.asarray(px)[:, None] - F.cx_scr.to_numpy()[None, :]
    dy = np.asarray(py)[:, None] - F.cy_scr.to_numpy()[None, :]
    u = dx*np.cos(t) + dy*np.sin(t)
    v = -dx*np.sin(t) + dy*np.cos(t)
    Q = (u/F.ell_a.to_numpy()[None, :])**2 + (v/F.ell_b.to_numpy()[None, :])**2
    j = Q.argmin(1)
    return Q[np.arange(len(Q)), j], j


def wilcoxon_p(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(b) < 5 or np.all(b == 0): return np.nan
    return float(stats.wilcoxon(b).pvalue)


# ============================================================ provenance
def sha256(path, cache):
    st = os.stat(path)
    k = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if k in cache: return cache[k]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    cache[k] = h.hexdigest()
    return cache[k]


def stage_provenance(root, out):
    say("\n=== provenance ===")
    cache_f = out/"hash_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    files = (sorted(glob.glob(str(root/"eyedata"/"**"/"*.asc"), recursive=True)) +
             sorted(glob.glob(str(root/"ImageTrials"/"**"/"*.*"), recursive=True)))
    files = [f for f in files if Path(f).is_file()
             and Path(f).suffix.lower() in (".asc", ".jpg", ".jpeg", ".png", ".bmp")]
    inv = []
    for i, f in enumerate(files, 1):
        inv.append(dict(path=str(Path(f).relative_to(root)),
                        bytes=os.path.getsize(f), sha256=sha256(f, cache)))
        if i % 200 == 0: say(f"  hashed {i}/{len(files)}")
    cache_f.write_text(json.dumps(cache))
    scripts = {s: sha256(str(root/s), {}) for s in
               ["trajextract.py", "facelabel.py", "ou_visits.py", "OU_field.py",
                "reproduce.py"] if (root/s).exists()}
    vers = dict(python=sys.version.split()[0], platform=platform.platform())
    for m in ["numpy", "pandas", "scipy", "statsmodels", "sklearn",
              "matplotlib", "torch", "face_alignment"]:
        try: vers[m] = __import__(m).__version__
        except Exception: vers[m] = "unavailable"
    rec = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               params=PARAMS, seeds=SEEDS, scripts=scripts, versions=vers,
               n_inputs=len(inv))
    (out/"provenance.json").write_text(json.dumps(rec, indent=2, default=str))
    pd.DataFrame(inv).to_csv(out/"inputs.csv", index=False)
    say(f"  {len(inv)} input files, {len(scripts)} scripts -> provenance.json, inputs.csv")


# ============================================================ extract / faces
def stage_extract(root, force):
    say("\n=== extract (trajextract.py) ===")
    n = 0
    for pid, ses, f in find_asc(root):
        ext = root/"output"/"extracted"/f"ses-{ses}"
        if not force and list(ext.glob(f"{pid}_ses-{ses}_samples.*")):
            continue
        sh([sys.executable, root/"trajextract.py", "--root", root, "--asc", f,
            "--outdir", ext], f"{pid}/{ses}"); n += 1
    say(f"  extracted {n}")


def stage_faces(root, force):
    say("\n=== faces (facelabel.py) ===")
    sets = {}
    if (root/"ImageTrials"/"IAPS").is_dir():
        sets["IAPS"] = root/"ImageTrials"/"IAPS"
    for d in sorted((root/"ImageTrials"/"ABMimages").glob("*_trials4experiment")):
        sets[d.name.replace("_trials4experiment", "")] = d
    for name, d in sets.items():
        od = root/"output"/"faces"/name
        if (od/"face_bounds.csv").exists() and not force: continue
        sh([sys.executable, root/"facelabel.py", "--imgdir", d, "--outdir", od,
            "--no-overlay"], name)


def load_faces(root):
    fs = glob.glob(str(root/"output"/"faces"/"*"/"face_bounds.csv"))
    if not fs: return pd.DataFrame()
    # IAPS images also sit in several ABM folders; the detector is
    # deterministic, so duplicates are dropped rather than averaged
    return (pd.concat([pd.read_csv(f, dtype={"stem": str}) for f in fs],
                      ignore_index=True).drop_duplicates(["stem", "face_id"]))


# ================================================================= samples
def stage_samples(root, force, ov):
    """§4.1: ĥ_im = T_im^(-1/6) σ̂_im per participant × image, over the samples
    the KDE is actually built from (ou_visits.clean_mask, same minimums), and
    ĥ_M their mean.  The same pass records the shape statistics reported in
    §5: robust Silverman (σ̂ from the MAD) and increment kurtosis."""
    say("\n=== samples (§4.1 bandwidths, §5 shape) ===")
    f = root/"output"/"bandwidth_table.csv"
    if f.exists() and not force:
        B = pd.read_csv(f)
        # tables written before this script lack the shape statistics; they
        # are borrowed from kde_shape.csv rather than recomputing the pass
        ks = root/"output"/"validation_ok"/"kde_shape.csv"
        for c in ["h_silv_robust", "kurt_inc"]:
            if c not in B.columns:
                if ks.exists():
                    K = pd.read_csv(ks)
                    B = B.merge(K[["pid","ses","image",c]], on=["pid","ses","image"],
                                how="left")
                else:
                    B[c] = np.nan
        say(f"  reusing {f.name} ({len(B)} image-sessions)"); return B
    rows = []
    for pid, ses, _ in find_asc(root):
        ext = root/"output"/"extracted"/f"ses-{ses}"
        sf = list(ext.glob(f"{pid}_ses-{ses}_samples.*"))
        if not sf: continue
        S = pd.read_parquet(sf[0]) if sf[0].suffix == ".parquet" else pd.read_csv(sf[0])
        cf = ext/f"{pid}_ses-{ses}_saccades.csv"
        C = pd.read_csv(cf) if cf.exists() else pd.DataFrame()
        ST = S[(S.phase == "stimulus") & (~S.practice.astype(bool))]
        for img, gi in ST.groupby("image"):
            xs, incs = [], []
            for _, gt in gi.groupby("trial_i"):
                gt = gt.sort_values("t_ms")
                t = gt.t_ms.to_numpy(float)
                x = gt.x.to_numpy(float); y = gt.y.to_numpy(float)
                m = ov.clean_mask(t, x, y, C)
                if m.sum() < PARAMS["min_samples_trial"]: continue
                X = np.column_stack([x[m], y[m]]); tt = t[m]
                xs.append(X)
                adj = np.diff(tt) <= PARAMS["sigma2_adj_ms"]
                d = np.diff(X, axis=0)[adj]
                incs.append(np.hypot(d[:, 0], d[:, 1]))
            if not xs: continue
            X = np.vstack(xs)
            if len(X) < PARAMS["min_samples_image"]: continue
            sd = X.std(0, ddof=1)
            rsd = 1.4826*np.median(np.abs(X - np.median(X, 0)), 0)
            inc = np.concatenate(incs) if incs else np.array([])
            rows.append(dict(
                pid=pid, ses=f"ses-{ses}", image=str(img),
                stem=str(img).rsplit(".", 1)[0], m=len(X),
                h_silv=ov.silverman_h(X),
                h_silv_robust=float(np.sqrt(max(rsd[0]*rsd[1], 1e-12)) *
                                    len(X)**(-1/6)),
                kurt_inc=(float(stats.kurtosis(inc, fisher=False))
                          if len(inc) > 50 else np.nan)))
        say(f"  {pid}/ses-{ses}")
    B = pd.DataFrame(rows); B.to_csv(f, index=False)
    say(f"  ĥ_M = {B.h_silv.mean():.3f} px over {len(B)} image-sessions")
    return B


# ==================================================================== runs
def stage_runs(root, B, force):
    """§4.1: the three bandwidths.  Each run re-estimates p̂, the basins, Λ_k,
    membership and visits from scratch; nothing is shared between them."""
    say("\n=== runs (ou_visits.py) ===")
    CONDS["hmean"] = ["--kde-mode", "fixed", "--kde-px", f"{B.h_silv.mean():.3f}"]
    fdirs = sorted((root/"output"/"faces").glob("*"))
    fargs = sum((["--faces", d] for d in fdirs), [])
    for pid, ses, _ in find_asc(root):
        for c, flags in CONDS.items():
            od = root/"output"/c/f"{pid}_ses-{ses}"
            vf = od/"visits.csv"
            if vf.exists() and not force:
                if "th1" in pd.read_csv(vf, nrows=1).columns: continue
                say(f"  {vf} lacks Λ_k eigenvalues — regenerating")
            sh([sys.executable, root/"ou_visits.py", "--root", root, "--pid", pid,
                "--ses", ses, "--outdir", od] + fargs + flags, f"{c} {pid}/{ses}")


def load_visits(root, cond):
    fs = glob.glob(str(root/"output"/cond/"*"/"visits.csv"))
    V = pd.concat([pd.read_csv(f, dtype={"stem": str}) for f in fs],
                  ignore_index=True)
    if "th1" not in V.columns or V.th1.isna().any():
        raise SystemExit(f"{cond}: visits.csv without Λ_k eigenvalues; "
                         f"ou_visits.py is not the version the document uses")
    return V


# ==================================================================== prep
def stage_prep(root, force):
    """Attach to every h = 26 visit: face identity (argmin_v Q_v(μ_k)), depth
    q = min_v Q_v(μ_k), and the trial's condition.  §3 'Face regions' and §4.3."""
    say("\n=== prep ===")
    f = root/"output"/"visits_full.csv"
    if f.exists() and not force:
        V = pd.read_csv(f, dtype={"stem": str})
        if {"face_id", "face_q", "condition", "th1"} <= set(V.columns):
            # stimulus timing is joined in memory if an older file lacks it
            if not {"stim_on_ms", "stim_off_ms"} <= set(V.columns):
                T = []
                for tf in glob.glob(str(root/"output"/"extracted"/"ses-*"/"*_trials.csv")):
                    t = pd.read_csv(tf); nm = Path(tf).name
                    t["pid"] = nm.split("_ses-")[0]
                    t["ses"] = "ses-" + nm.split("_ses-")[1][:2]
                    T.append(t[["pid","ses","trial_i","stim_on_ms","stim_off_ms"]])
                V = V.merge(pd.concat(T), on=["pid","ses","trial_i"], how="left")
            say(f"  reusing {f.name}"); return V
        say(f"  {f.name} predates this script — rebuilding")
    V = load_visits(root, "h26"); FB = load_faces(root)
    Bk = V.drop_duplicates(KEY + ["basin"])[KEY + ["basin", "mu_x", "mu_y"]]
    Bk = Bk.reset_index(drop=True)
    fid = np.full(len(Bk), -1); fq = np.full(len(Bk), np.nan)
    faces = {s: g for s, g in FB.groupby("stem")}
    for s, g in Bk.groupby("stem"):
        F = faces.get(s)
        if F is None: continue
        q, j = face_q(g.mu_x.values, g.mu_y.values, F)
        idx = g.index.to_numpy()
        fid[idx] = np.where(q <= 1.0, F.face_id.to_numpy()[j], -1)
        fq[idx] = q
    Bk["face_id"] = fid; Bk["face_q"] = fq
    V = V.merge(Bk[KEY + ["basin", "face_id", "face_q"]], on=KEY + ["basin"],
                how="left")
    T = []
    for tf in glob.glob(str(root/"output"/"extracted"/"ses-*"/"*_trials.csv")):
        t = pd.read_csv(tf); nm = Path(tf).name
        t["pid"] = nm.split("_ses-")[0]; t["ses"] = "ses-" + nm.split("_ses-")[1][:2]
        T.append(t[["pid", "ses", "trial_i", "var_condition",
                    "stim_on_ms", "stim_off_ms"]])
    T = pd.concat(T, ignore_index=True).rename(columns={"var_condition": "condition"})
    V = V.merge(T, on=["pid", "ses", "trial_i"], how="left")
    agree = ((V.face_id >= 0) == V.on_face.astype(bool)).mean()
    say(f"  {len(V)} visits; face identity agrees with ou_visits on_face "
        f"for {agree:.4f} of visits")
    V.to_csv(f, index=False)
    return V


# ============================================================ validation
class Report:
    def __init__(self):
        self.rows = []
        self.per_session = []      # (effect, coefficient) for the forest plot
        self.nulls = []            # (test, observed, null draw) for the null plots
    def add(self, section, name, value, detail=""):
        self.rows.append(dict(section=section, quantity=name, value=value,
                              detail=detail))
        v = f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
        say(f"  [{section}] {name:52} {v:>12}   {detail}")


def derived(V):
    V = V[V.dwell_ms > 0].copy()
    V["unit"] = V.pid + "/" + V.ses
    V["imgses"] = V.unit + "/" + V.stem
    V["onf"] = (V.face_id >= 0).astype(int)
    V["ldwell"] = np.log(V.dwell_ms)
    V["ltheta"] = np.log(V.theta.clip(1e-9))
    V["depth"] = np.where(V.onf == 1, 1 - V.face_q.clip(0, 1), 0.0)
    V["val"] = V.condition.str[0]
    V["iaps"] = V.stem.str.fullmatch(r"\d{4}")
    V["cohort"] = V.pid.str[0]
    V["t_onset"] = (V.entry_ms - V.stim_on_ms)/1000.0
    V["truncated"] = V.exit_ms >= V.stim_off_ms - PARAMS["truncation_ms"]
    return V


def basin_table(V):
    return (V.groupby(KEY + ["basin"])
              .agg(mu_x=("mu_x", "first"), mu_y=("mu_y", "first"),
                   theta=("theta", "first"), basin_px=("basin_px", "first"),
                   sigma2=("sigma2", "first"), on_face=("on_face", "first"),
                   dwell=("dwell_ms", "sum"))
              .reset_index())


# ---------------------------------------------------------- §4.1 bandwidth
def v_bandwidth(root, R, Bw):
    say("\n--- §4.1 bandwidth ---")
    Vs = {c: load_visits(root, c) for c in ["h26", "hsilv", "hmean"]}
    R.add("4.1", "ĥ_M (mean Silverman, px)", Bw.h_silv.mean(),
          f"median {Bw.h_silv.median():.1f}")
    R.add("4.1", "robust Silverman, median (px)", Bw.h_silv_robust.median(),
          f"IQR {Bw.h_silv_robust.quantile(.25):.1f}-{Bw.h_silv_robust.quantile(.75):.1f}")

    # ordering of sessions under each h: Spearman ρ on four summaries
    S = []
    for c, V in Vs.items():
        V = V[V.dwell_ms > 0]
        g = V.groupby(["pid", "ses"])
        S.append(pd.DataFrame(dict(
            cond=c,
            basins=g.apply(lambda d: d.groupby("stem").basin.nunique().mean()),
            on_face=g.on_face.mean(), dwell=g.dwell_ms.median(),
            theta=g.theta.median())).reset_index())
    S = pd.concat(S)
    for m in ["basins", "on_face", "dwell", "theta"]:
        W = S.pivot_table(index=["pid", "ses"], columns="cond", values=m)
        for a, b in [("h26", "hsilv"), ("h26", "hmean"), ("hsilv", "hmean")]:
            d = W[[a, b]].dropna()
            R.add("4.1", f"Spearman ρ {m} {a}~{b}",
                  float(d[a].corr(d[b], method="spearman")), f"n={len(d)}")

    # correspondence and persistence (§4.3 stability)
    Bk = {c: basin_table(V) for c, V in Vs.items()}
    idx = {c: {k: g for k, g in Bk[c].groupby(KEY)} for c in Bk}
    tol = PARAMS["match_px"]

    def matched(P, Qi):
        hit = tot = 0
        for k, gp in P.groupby(KEY):
            gq = Qi.get(k); tot += len(gp)
            if gq is None or gq.empty: continue
            d = np.hypot(gp.mu_x.values[:, None]-gq.mu_x.values[None, :],
                         gp.mu_y.values[:, None]-gq.mu_y.values[None, :])
            hit += int((d.min(1) <= tol).sum())
        return hit/max(tot, 1)
    for a, b in [("h26", "hsilv"), ("h26", "hmean"), ("hsilv", "hmean")]:
        R.add("4.1", f"basins matched within 26 px {a}->{b}", matched(Bk[a], idx[b]))

    rows = []
    for k, gp in Bk["h26"].groupby(KEY):
        n = np.ones(len(gp), int)
        for c in ["hsilv", "hmean"]:
            gq = idx[c].get(k)
            if gq is None or gq.empty: continue
            d = np.hypot(gp.mu_x.values[:, None]-gq.mu_x.values[None, :],
                         gp.mu_y.values[:, None]-gq.mu_y.values[None, :])
            n += (d.min(1) <= tol).astype(int)
        rows.append(pd.DataFrame(dict(persist=n, on_face=gp.on_face.values)))
    P = pd.concat(rows)
    for p, g in P.groupby("persist"):
        R.add("5", f"on-face rate, basins found by {p} bandwidth(s)",
              float(g.on_face.mean()), f"n={len(g)}")

    # adjusted Rand index between labellings of the same trial
    def ari(a, b):
        ua, ia = np.unique(a, return_inverse=True)
        ub, ib = np.unique(b, return_inverse=True)
        Cm = np.zeros((len(ua), len(ub))); np.add.at(Cm, (ia, ib), 1)
        n = len(a); s_ij = (Cm*(Cm-1)/2).sum()
        ai = Cm.sum(1); bj = Cm.sum(0)
        si = (ai*(ai-1)/2).sum(); sj = (bj*(bj-1)/2).sum(); tot = n*(n-1)/2
        exp = si*sj/tot if tot else 0; mx = (si+sj)/2
        return (s_ij-exp)/(mx-exp) if mx != exp else np.nan
    tr = {c: {k: g for k, g in V.sort_values("entry_ms").groupby(KEY + ["trial_i"])}
          for c, V in Vs.items()}
    common = set.intersection(*[set(t) for t in tr.values()])
    out = {p: [] for p in [("h26", "hsilv"), ("h26", "hmean"), ("hsilv", "hmean")]}
    for k in common:
        t0 = min(tr[c][k].entry_ms.min() for c in tr)
        t1 = max(tr[c][k].exit_ms.max() for c in tr)
        grid = np.arange(t0, t1, PARAMS["ari_grid_ms"])
        if len(grid) < 20: continue
        lab = {}
        for c in tr:
            l = np.full(len(grid), -1)
            for r in tr[c][k].itertuples():
                l[(grid >= r.entry_ms) & (grid <= r.exit_ms)] = r.basin
            lab[c] = l
        for p in out: out[p].append(ari(lab[p[0]], lab[p[1]]))
    for p, v in out.items():
        R.add("5", f"adjusted Rand index {p[0]}~{p[1]}, median",
              float(np.nanmedian(v)), f"{len(v)} trials")


# ------------------------------------------------------- §4.2 face capture
def v_face_capture(root, R, V):
    say("\n--- §4.2 face capture ---")
    FB = load_faces(root); faces = {s: g for s, g in FB.groupby("stem")}
    Bk = V.drop_duplicates(KEY + ["basin"]).copy()
    Bk = Bk[Bk.stem.isin(faces)]
    pts = {k: g[["mu_x", "mu_y"]].to_numpy() for k, g in Bk.groupby(["unit", "stem"])}
    by_unit = Bk.groupby("unit").stem.unique().to_dict()

    gx = np.arange(PARAMS["grid_px"]/2, PARAMS["screen_px"][0], PARAMS["grid_px"])
    gy = np.arange(PARAMS["grid_px"]/2, PARAMS["screen_px"][1], PARAMS["grid_px"])
    GX, GY = np.meshgrid(gx, gy); G = np.column_stack([GX.ravel(), GY.ravel()])
    ox, oy = PARAMS["stim_offset_px"]; sw, shh = PARAMS["stim_px"]
    G = G[(G[:, 0] >= ox) & (G[:, 0] <= ox+sw) & (G[:, 1] >= oy) & (G[:, 1] <= oy+shh)]

    def hits(P, F):
        q, _ = face_q(P[:, 0], P[:, 1], F); return int((q <= 1).sum())

    ru = np.random.default_rng(SEEDS["uniform"])
    ra = np.random.default_rng(SEEDS["alt"])
    rows = []
    for (u, s), P in pts.items():
        F = faces[s]; K = len(P)
        O = hits(P, F)
        U = np.mean([hits(G[ru.choice(len(G), K, replace=False)], F)
                     for _ in range(PARAMS["n_uniform"])])
        alt = [t for t in by_unit[u] if t != s]
        A = AK = np.nan
        if alt:
            pk = [alt[ra.integers(len(alt))] for _ in range(PARAMS["n_alt"])]
            A = np.mean([hits(pts[(u, t)], F) for t in pk])
            AK = np.mean([len(pts[(u, t)]) for t in pk])
        rows.append(dict(unit=u, stem=s, iaps=s.isdigit(), K=K, O=O, U=U,
                         A=A, AK=AK))
    D = pd.DataFrame(rows)
    D.to_csv(root/"output"/"reproduce"/"face_capture.csv", index=False)
    for lab, g in [("IAPS", D[D.iaps]), ("ABM", D[~D.iaps])]:
        R.add("4.2", f"O/U {lab}", g.O.sum()/g.U.sum(), f"{len(g)} image-sessions")

    # O/A_U on ABM: A rescaled per image-session to that image's basin count,
    # summed per session, then aggregated; session is the unit of the test
    M = D[~D.iaps].dropna(subset=["A"]).copy()
    M["Ae"] = M.A*M.K/M.AK
    Ps = M.groupby("unit").agg(O=("O", "sum"), A=("Ae", "sum")).reset_index()
    Ps["r"] = Ps.O/Ps.A
    rb = np.random.default_rng(SEEDS["boot"])
    bs = []
    for _ in range(PARAMS["n_boot"]):
        s_ = Ps.iloc[rb.integers(len(Ps), size=len(Ps))]
        bs.append(s_.O.sum()/s_.A.sum())
    w = stats.wilcoxon(Ps.O, Ps.A, alternative="greater")
    R.add("4.2", "O/A_U ABM", Ps.O.sum()/Ps.A.sum(),
          f"95% CI {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}")
    R.add("4.2", "O/A_U ABM per-session median", float(Ps.r.median()),
          f"IQR {Ps.r.quantile(.25):.2f}-{Ps.r.quantile(.75):.2f}; "
          f"{int((Ps.r > 1).sum())}/{len(Ps)} > 1")
    R.add("4.2", "O/A_U ABM Wilcoxon W", float(w.statistic), f"p={w.pvalue:.2e}")
    Ps.to_csv(root/"output"/"reproduce"/"capture_abm_sessions.csv", index=False)

    # A_iC on IAPS, per participant: others' hit RATE on the same image applied
    # to this participant's basin count.  Summed over everyone it is 1 by
    # construction, so only the per-session distribution is reported.
    I = Bk[Bk.iaps].copy(); I["hit"] = (I.face_id >= 0).astype(int)
    Cn = I.groupby(["unit", "stem"]).agg(K=("hit", "size"), h=("hit", "sum")).reset_index()
    r2 = []
    for s, g in Cn.groupby("stem"):
        if len(g) < 3: continue
        Ks, hs = g.K.to_numpy(), g.h.to_numpy()
        for i in range(len(g)):
            o = np.delete(np.arange(len(g)), i)
            r2.append(dict(unit=g.unit.iloc[i], O=hs[i],
                           A=hs[o].sum()/max(Ks[o].sum(), 1)*Ks[i]))
    Pp = pd.DataFrame(r2).groupby("unit").agg(O=("O", "sum"), A=("A", "sum")).reset_index()
    Pp["r"] = Pp.O/Pp.A
    Pp.to_csv(root/"output"/"reproduce"/"capture_iaps_sessions.csv", index=False)
    chi = float(((Pp.O-Pp.A)**2/Pp.A).sum()); df = len(Pp)-1
    R.add("4.2", "O/A_C IAPS per-session median", float(Pp.r.median()),
          f"IQR {Pp.r.quantile(.25):.2f}-{Pp.r.quantile(.75):.2f}")
    R.add("4.2", "O/A_C IAPS heterogeneity χ²", chi,
          f"df={df} p={1-stats.chi2.cdf(chi, df):.2e}")
    Pp["pid"] = Pp.unit.str.split("/").str[0]
    Pp["t"] = Pp.unit.str.split("/").str[1].map(
        {"ses-05": "early", "ses-06": "early", "ses-16": "late", "ses-17": "late"})
    Wp = Pp.pivot_table(index="pid", columns="t", values="r").dropna()
    if len(Wp) >= 4:
        rho = stats.spearmanr(Wp.early, Wp.late)
        R.add("4.2", "O/A_C IAPS stability across sessions ρ",
              float(rho.correlation), f"n={len(Wp)} p={rho.pvalue:.3f}")


# -------------------------------------------------------------- §5 effects
def per_session(df, formula, term, min_n=30):
    import statsmodels.formula.api as smf
    b = []
    for _, g in df.groupby("unit"):
        if len(g) < min_n: continue
        try:
            m = smf.ols(formula, data=g).fit()
            if term in m.params and np.isfinite(m.params[term]): b.append(m.params[term])
        except Exception:
            pass
    return np.array(b)


def effect(R, name, df, formula, term):
    """§4.3 'Dwell model': fitted within each session, the coefficients tested
    against zero; the visit-level fit with image-session clustering alongside."""
    import statsmodels.formula.api as smf
    b = per_session(df, formula, term)
    if len(b) < 5:
        R.add("5", name, np.nan, f"skipped: {len(b)} sessions estimable"); return
    try:
        mc = smf.ols(formula + " + C(pid)", data=df).fit()
        g = pd.factorize(df.loc[mc.model.data.row_labels, "imgses"])[0]
        mc = smf.ols(formula + " + C(pid)", data=df).fit(cov_type="cluster",
                                                         cov_kwds={"groups": g})
    except Exception as e:
        R.add("5", name, np.nan, f"skipped: {type(e).__name__}"); return
    k = int((np.sign(b) == np.sign(b.mean())).sum())
    R.per_session += [(name, float(x)) for x in b]
    R.add("5", name, 100*(np.exp(b.mean())-1),
          f"% dwell; {k}/{len(b)} sessions; p={wilcoxon_p(b):.1e}; "
          f"clustered t={mc.tvalues[term]:+.1f}")


def v_effects(root, R, V):
    say("\n--- §5 dwell effects ---")
    FB = load_faces(root)
    V = V.merge(FB[["stem", "face_id", "ell_area"]], on=["stem", "face_id"], how="left")
    V["larea"] = np.log(V.ell_area.clip(1))
    F0 = V[V.re == 0]; W = V[V.re <= 6]; Wc = W[~W.truncated]
    long_ = PARAMS["brief_ms"]

    effect(R, "face effect, first entry", F0, "ldwell ~ onf + ltheta", "onf")
    effect(R, "face effect, first entry, IAPS", F0[F0.iaps], "ldwell ~ onf + ltheta", "onf")
    effect(R, "face effect, first entry, dwell >= 30 ms", F0[F0.dwell_ms >= long_],
           "ldwell ~ onf + ltheta", "onf")
    effect(R, "centring (depth), on-face first entry", F0[F0.onf == 1],
           "ldwell ~ depth + ltheta", "depth")
    effect(R, "re-entry, per return", W, "ldwell ~ onf*re + ltheta", "re")
    effect(R, "re-entry, truncated removed + time since onset", Wc,
           "ldwell ~ onf*re + t_onset + ltheta", "re")
    effect(R, "re-entry, IAPS", Wc[Wc.iaps], "ldwell ~ onf*re + t_onset + ltheta", "re")
    effect(R, "re-entry, dwell >= 30 ms", Wc[Wc.dwell_ms >= long_],
           "ldwell ~ onf*re + t_onset + ltheta", "re")
    effect(R, "faces slow the decline (onf:re)", Wc,
           "ldwell ~ onf*re + t_onset + ltheta", "onf:re")

    On = V[(V.onf == 1) & V.val.isin(["N", "P"]) & (V.re <= 6)].copy()
    On["pos"] = (On.val == "P").astype(int)
    effect(R, "P vs N on faces", On, "ldwell ~ pos + re + ltheta", "pos")
    effect(R, "P vs N on faces, + log face area", On.dropna(subset=["larea"]),
           "ldwell ~ pos + larea + re + ltheta", "pos")
    effect(R, "P vs N on faces, IAPS", On[On.iaps].dropna(subset=["larea"]),
           "ldwell ~ pos + larea + re + ltheta", "pos")
    Of = V[(V.onf == 0) & V.val.isin(["N", "P"]) & (V.re <= 6)].copy()
    Of["pos"] = (Of.val == "P").astype(int)
    effect(R, "P vs N off faces", Of, "ldwell ~ pos + re + ltheta", "pos")

    # centring as the raw bins of §5
    G = V[V.onf == 1]
    bins = G.groupby(pd.cut(G.face_q, [0, .2, .4, .6, .8, 1.0])).dwell_ms.median()
    R.add("5", "on-face median dwell, q in (0,0.2]", float(bins.iloc[0]), "ms")
    R.add("5", "on-face median dwell, q in (0.8,1]", float(bins.iloc[-1]), "ms")

    # re-entry table: brief fraction and sustained median, by region
    T = W.assign(brief=W.dwell_ms < long_).groupby(["onf", "re"]).agg(
        n=("dwell_ms", "size"), brief=("brief", "mean"),
        sustained=("dwell_ms", lambda s: s[s >= long_].median())).reset_index()
    T.to_csv(root/"output"/"reproduce"/"reentry_table.csv", index=False)
    say("  re-entry table -> reentry_table.csv"); say(T.round(3).to_string(index=False))

    # valence shape: per participant × valence, face basins above the floor,
    # centred on the median and scaled by the IQR, Anderson–Darling k-sample
    Bf = (V[V.onf == 1].groupby(KEY + ["basin"])
          .agg(val=("val", "first"), basin_px=("basin_px", "first"),
               dwell=("dwell_ms", "median")).reset_index())
    Bf = Bf[(Bf.basin_px > PARAMS["floor_px"]) & Bf.val.isin(["N", "P"])]
    groups = []
    for _, g in Bf.groupby(["pid", "ses", "val"]):
        if len(g) < 12: continue
        x = np.log(g.dwell.to_numpy())
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr > 0: groups.append((x-np.median(x))/iqr)
    ad = stats.anderson_ksamp(groups)
    R.add("5", "valence: standardised dwell shape, Anderson–Darling p",
          float(ad.pvalue), f"{len(groups)} participant × valence cells")


def v_attendance(root, R, V):
    """§4.3 'Attendance': every participant–face pair, whether or not a basin
    landed on the face."""
    say("\n--- §5 attendance ---")
    import statsmodels.formula.api as smf
    FB = load_faces(root)
    seen = V.drop_duplicates(KEY)[KEY + ["val"]]
    A = seen.merge(FB[["stem", "face_id", "ell_area", "cx_scr", "cy_scr"]],
                   on="stem", how="inner")
    got = (V[V.onf == 1].groupby(KEY + ["face_id"]).size().rename("nvis").reset_index())
    A = A.merge(got, on=KEY + ["face_id"], how="left").fillna({"nvis": 0})
    A["att"] = (A.nvis > 0).astype(int)
    A["nfaces"] = A.groupby(KEY).face_id.transform("size")
    A["rank"] = A.groupby(KEY).ell_area.rank(ascending=False, method="min")
    A["larea"] = np.log10(A.ell_area.clip(1))
    A["lnf"] = np.log(A.nfaces)
    A["lecc"] = np.log(np.hypot(A.cx_scr-640, A.cy_scr-512).clip(1))
    A["unit"] = A.pid + "/" + A.ses
    R.add("5", "faces attended, overall", float(A.att.mean()), f"{len(A)} pairs")
    (A.groupby(pd.qcut(A.ell_area, 10, duplicates="drop"), observed=True)
       .agg(area=("ell_area", "median"), attended=("att", "mean"), n=("att", "size"))
       .reset_index(drop=True)
       .to_csv(root/"output"/"reproduce"/"attendance_by_area.csv", index=False))
    R.add("5", "faces attended, single-face images", float(A[A.nfaces == 1].att.mean()))
    R.add("5", "faces attended, multi-face images", float(A[A.nfaces > 1].att.mean()))
    R.add("5", "faces attended, largest face in multi-face image",
          float(A[(A.nfaces > 1) & (A["rank"] == 1)].att.mean()))
    mc = smf.logit("att ~ larea + lnf + lecc + C(val)", data=A).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(A.stem)[0]})
    b = []
    for _, g in A.groupby("unit"):
        if g.att.nunique() < 2 or len(g) < 40: continue
        try: b.append(smf.logit("att ~ larea", data=g).fit(disp=0).params["larea"])
        except Exception: pass
    b = np.array(b)
    R.add("5", "attendance ~ log10 face area, per-session β", float(b.mean()),
          f"{int((b > 0).sum())}/{len(b)} positive; clustered t={mc.tvalues['larea']:+.1f}")


# -------------------------------------------------------- §5 model checks
def _blocks(c):
    ch = np.flatnonzero(np.diff(c))+1
    return np.r_[0, ch], np.r_[ch, len(c)]


def _between(val, lab, lo, hi):
    num = den = 0.0
    for a, b in zip(lo, hi):
        if b-a < 2: continue
        v = val[a:b]; u, inv = np.unique(lab[a:b], return_inverse=True)
        if len(u) < 2: continue
        m = v.mean(); cnt = np.bincount(inv); sm = np.bincount(inv, weights=v)
        num += (cnt*(sm/cnt-m)**2).sum(); den += ((v-m)**2).sum()
    return num/den if den > 0 else np.nan


def v_model(root, R, V, Bw):
    say("\n--- §5 model assumptions ---")
    # J = 0: asymmetry of transition counts against a within-trial shuffle
    def asym(chains):
        a = s = 0.0
        for b in chains:
            k = int(b.max())+1; N = np.zeros((k, k))
            np.add.at(N, (b[:-1], b[1:]), 1)
            a += np.abs(N-N.T).sum(); s += np.abs(N+N.T).sum()
        return a/s
    Vs = V.sort_values(KEY + ["trial_i", "entry_ms"])
    ch = [g.basin.to_numpy() for _, g in Vs.groupby(KEY + ["trial_i"], sort=False)]
    ch = [c for c in ch if len(c) >= 2]
    rng = np.random.default_rng(SEEDS["jperm"])
    obs = asym(ch)
    null = np.array([asym([rng.permutation(c) for c in ch])
                     for _ in range(PARAMS["n_perm"])])
    R.nulls += [("transition asymmetry", obs, float(x)) for x in null]
    R.add("5", "transition asymmetry A", obs,
          f"null {null.mean():.3f}±{null.std():.3f}, z={(obs-null.mean())/null.std():+.2f}")

    # differences between faces: basin-level permutation of face labels
    F = V[V.onf == 1]
    F = F[F.groupby(KEY).face_id.transform("nunique") > 1]
    Bf = (F.groupby(KEY + ["basin"])
          .agg(face_id=("face_id", "first"), theta=("theta", "first"),
               face_q=("face_q", "first"), dwell=("dwell_ms", "median"),
               val=("val", "first")).reset_index())
    rng = np.random.default_rng(SEEDS["facediff"])

    def fd(B, col, tag):
        B = B[B.groupby(KEY).face_id.transform("nunique") > 1]
        B = B.sort_values(KEY).reset_index(drop=True)
        lo, hi = _blocks(pd.factorize(B.pid+"|"+B.ses+"|"+B.stem)[0])
        lab = B.face_id.to_numpy()
        val = np.log(np.clip(B[col].to_numpy(float), 1e-9, None))
        o = _between(val, lab, lo, hi); nl = np.empty(PARAMS["n_perm"])
        for i in range(PARAMS["n_perm"]):
            L = lab.copy()
            for a, b in zip(lo, hi):
                if b-a > 1: L[a:b] = rng.permutation(lab[a:b])
            nl[i] = _between(val, L, lo, hi)
        R.nulls += [(f"between-face {col}{tag}", o, float(x)) for x in nl]
        R.add("5", f"between-face {col}{tag}", (o-nl.mean())/nl.std(),
              f"z; observed {o:.3f} null {nl.mean():.3f}; {len(B)} basins")
    for col in ["dwell", "theta", "face_q"]: fd(Bf, col, "")
    for v in ["N", "P"]: fd(Bf[Bf.val == v], "dwell", f" [{v}]")

    # pull does not predict dwell
    import statsmodels.formula.api as smf
    tau = V.theta*V.dwell_ms/1000.0
    R.add("5", "θ·Δ_dwell, median", float(tau.median()), "OU predicts O(1)")
    F0 = V[V.re == 0]
    R.add("5", "corr(log θ, log Δ_dwell), first entry",
          float(np.corrcoef(F0.ltheta, F0.ldwell)[0, 1]), "OU predicts -1")
    m = smf.ols("ldwell ~ ltheta + C(pid)", data=F0).fit()
    R.add("5", "log–log slope, first entry", float(m.params["ltheta"]), "OU predicts -1")
    Bk = V.drop_duplicates(KEY + ["basin"])
    Bk = Bk[Bk.basin_px > PARAMS["floor_px"]]
    lam = Bk.theta/(Bk.sigma2/2)
    R.add("5", "s.d. of log λ, basins above the floor", float(np.log(lam).std()))
    R.add("5", "share of basins at the smoothing floor",
          float((V.drop_duplicates(KEY + ["basin"]).basin_px <= PARAMS["floor_px"]).mean()))
    R.add("5", "increment kurtosis, median", float(Bw.kurt_inc.median()), "Gaussian = 3")
    s2 = V.drop_duplicates(KEY).groupby("cohort").sigma2.median()
    for c, v in s2.items():
        R.add("5", f"σ² median, cohort {c}", float(v), "px²/s")

    # IAPS consensus locations and their face rate
    I = V[V.iaps].drop_duplicates(KEY + ["basin"])
    I = I.assign(unit=I.pid+"/"+I.ses)
    out = []
    for _, g in I.groupby("stem"):
        P = g[["mu_x", "mu_y"]].to_numpy(); u = g.unit.to_numpy()
        f = g.on_face.to_numpy().astype(float)
        d = np.hypot(P[:, 0][:, None]-P[:, 0][None, :], P[:, 1][:, None]-P[:, 1][None, :])
        used = np.zeros(len(P), bool)
        while not used.all():
            best = None
            for i in np.where(~used)[0]:
                mem = np.where((d[i] <= PARAMS["match_px"]) & ~used)[0]
                n = len(np.unique(u[mem]))
                if best is None or n > best[0]: best = (n, mem)
            out.append((best[0], f[best[1]].mean())); used[best[1]] = True
    C = pd.DataFrame(out, columns=["n", "on_face"])
    R.add("5", "IAPS consensus, on-face rate, 17+ sessions",
          float(C[C.n >= 17].on_face.mean()), f"{int((C.n >= 17).sum())} locations")
    R.add("5", "IAPS consensus, on-face rate, 1 session",
          float(C[C.n == 1].on_face.mean()), f"{int((C.n == 1).sum())} locations")


# ================================================================= figures
def stage_figures(root, V, Bw):
    """One figure per part of the document, drawn only from what the
    validation stage wrote to output/reproduce/ plus the visits themselves.
    Nothing here is re-estimated."""
    say("\n=== figures ===")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = root/"output"/"reproduce"
    need = ["results.csv", "per_session.csv", "nulls.csv", "reentry_table.csv",
            "capture_abm_sessions.csv", "capture_iaps_sessions.csv",
            "attendance_by_area.csv"]
    miss = [n for n in need if not (out/n).exists()]
    if miss:
        say(f"  missing {miss}: run the validate stage first"); return
    R = pd.read_csv(out/"results.csv")
    Ps = pd.read_csv(out/"per_session.csv"); N = pd.read_csv(out/"nulls.csv")
    T = pd.read_csv(out/"reentry_table.csv")
    Ca = pd.read_csv(out/"capture_abm_sessions.csv")
    Ci = pd.read_csv(out/"capture_iaps_sessions.csv")
    At = pd.read_csv(out/"attendance_by_area.csv")
    val = lambda q: R.loc[R.quantity == q, "value"].astype(float)
    ON, OFF = "#cc3311", "#0077bb"

    def save(fig, name):
        fig.tight_layout()
        try: fig.savefig(out/name, dpi=140, bbox_inches="tight")
        except OSError:                     # OneDrive can refuse the write
            import shutil
            fig.savefig(f"/tmp/{name}", dpi=140, bbox_inches="tight")
            shutil.copy(f"/tmp/{name}", out/name)
        plt.close(fig); say(f"  {name}")

    # ---- §4.1 bandwidth -------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    rho = R[R.quantity.str.startswith("Spearman")].copy()
    rho["metric"] = rho.quantity.str.split().str[2]
    rho["pair"] = rho.quantity.str.split().str[3]
    piv = rho.pivot(index="metric", columns="pair", values="value").astype(float)
    piv.plot(kind="bar", ax=ax[0], rot=0)
    ax[0].axhline(1, color="k", lw=.8); ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("Spearman ρ across sessions")
    ax[0].set_title("session ordering under the three bandwidths"); ax[0].legend(fontsize=7)
    gr = np.linspace(0, 90, 300)
    for c, lab, col in [("h_silv", "Silverman $\\hat h_{im}$", OFF),
                        ("h_silv_robust", "robust scale", ON)]:
        v = Bw[c].dropna().to_numpy()
        if len(v) < 5: continue
        bw = 1.06*v.std()*len(v)**-.2
        ax[1].plot(gr, np.exp(-0.5*((gr[:, None]-v[None, :])/bw)**2).sum(1)
                   / (len(v)*bw*np.sqrt(2*np.pi)), color=col, lw=2,
                   label=f"{lab}, median {np.median(v):.1f}")
    ax[1].axvline(PARAMS["kde_px"], color="k", ls="--", label="h = 26")
    ax[1].set_xlabel("bandwidth (px)"); ax[1].set_title("per-image bandwidth")
    ax[1].legend(fontsize=8)
    pr = R[R.quantity.str.startswith("on-face rate, basins found by")]
    ax[2].bar(range(1, len(pr)+1), pr.value.astype(float), color=ON)
    ax[2].set_xticks(range(1, len(pr)+1))
    ax[2].set_xlabel("bandwidths that find the basin"); ax[2].set_ylabel("on-face rate")
    ax[2].set_title("persistent basins sit on faces")
    for a in ax: a.grid(alpha=.3)
    save(fig, "fig_4_1_bandwidth.png")

    # ---- §4.2 face capture ----------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ou = [float(val("O/U IAPS").iloc[0]), float(val("O/U ABM").iloc[0])]
    ax[0].bar(["IAPS", "ABM"], ou, color=[OFF, ON])
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_ylabel("$O/U$"); ax[0].set_title("basins on faces vs uniform grid points")
    ax[1].hist(Ca.r, bins=15, color=ON)
    ax[1].axvline(1, color="k", ls="--", lw=1.5)
    ax[1].axvline(Ca.O.sum()/Ca.A.sum(), color="k", lw=2,
                  label=f"aggregate {Ca.O.sum()/Ca.A.sum():.2f}")
    ax[1].set_xlabel("$O/A_U$ per session"); ax[1].set_title("ABM: vs own basins, other image")
    ax[1].legend(fontsize=8)
    Ci = Ci.sort_values("r")
    ax[2].errorbar(Ci.r, range(len(Ci)), xerr=1.96*np.sqrt(Ci.A)/Ci.A, fmt="o",
                   color=OFF, ecolor="#aaaaaa")
    ax[2].axvline(1, color="k", ls="--", lw=1.5)
    ax[2].set_yticks(range(len(Ci))); ax[2].set_yticklabels(Ci.unit, fontsize=6)
    ax[2].set_xlabel("$O_j/A_{C,j}$"); ax[2].set_title("IAPS: vs other viewers, same image")
    for a in ax: a.grid(alpha=.3)
    save(fig, "fig_4_2_face_capture.png")

    # ---- §5 dwell effects, per session ----------------------------------
    order = list(dict.fromkeys(Ps.effect))
    fig, ax = plt.subplots(figsize=(10, 0.45*len(order)+1.5))
    for y, e in enumerate(order[::-1]):
        b = Ps[Ps.effect == e].coef.to_numpy()
        m, se = b.mean(), b.std(ddof=1)/np.sqrt(len(b))
        mid = 100*(np.exp(m)-1)
        lo, hi = 100*(np.exp(m-1.96*se)-1), 100*(np.exp(m+1.96*se)-1)
        ax.errorbar(mid, y, xerr=[[mid-lo], [hi-mid]], fmt="o", color=OFF,
                    ecolor="#888888", capsize=3)
        ax.scatter(100*(np.exp(b)-1), np.full(len(b), y), s=6, color="#bbbbbb",
                   zorder=0)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1], fontsize=8)
    ax.set_xlabel("effect on dwell (%), per session; mean and 95% CI across sessions")
    ax.set_title("§5 dwell effects, estimated within each session")
    ax.grid(alpha=.3, axis="x")
    save(fig, "fig_5_effects.png")

    # ---- §5 re-entry split ----------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for o, col, lab in [(0, OFF, "off face"), (1, ON, "on face")]:
        g = T[T.onf == o]
        ax[0].plot(g.re, 100*g.brief, "o-", color=col, lw=2, label=lab)
        ax[1].plot(g.re, g.sustained, "o-", color=col, lw=2, label=lab)
    ax[0].set_xlabel("re-entry index $r$"); ax[0].set_ylabel("visits < 30 ms (%)")
    ax[0].set_title("brief grazes accumulate with re-entry")
    ax[1].set_xlabel("re-entry index $r$"); ax[1].set_ylabel("median dwell of visits ≥ 30 ms")
    ax[1].set_title("sustained dwell: flat off-face, decline on faces")
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=8)
    save(fig, "fig_5_reentry.png")

    # ---- §5 attendance and model assumptions ----------------------------
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    ax[0].plot(At.area, At.attended, "o-", color=ON, lw=2)
    ax[0].set_xscale("log"); ax[0].set_xlabel("face area (px²)")
    ax[0].set_ylabel("share attended"); ax[0].set_title("face size decides attendance")
    j = N[N.test == "transition asymmetry"]
    ax[1].hist(j.null, bins=40, color="#bbbbbb", label="null: order shuffled")
    ax[1].axvline(j.observed.iloc[0], color=ON, lw=2.5, label="observed")
    ax[1].set_xlabel("$A$"); ax[1].set_title("$J=0$: transition asymmetry")
    ax[1].legend(fontsize=8)
    bf = R[R.quantity.str.startswith("between-face")]
    ax[2].barh(bf.quantity.str.replace("between-face ", ""), bf.value.astype(float),
               color=[ON if v > 2 else "#bbbbbb" for v in bf.value.astype(float)])
    ax[2].axvline(2, color="k", ls="--", lw=1); ax[2].axvline(-2, color="k", ls="--", lw=1)
    ax[2].set_xlabel("z vs basin-level permutation")
    ax[2].set_title("do faces within an image differ?")
    F0 = V[V.re == 0].sample(min(20000, (V.re == 0).sum()), random_state=0)
    ax[3].scatter(np.log10(F0.theta.clip(1e-4)), np.log10(F0.dwell_ms),
                  s=1, alpha=.08, color="k")
    xl = np.array([-2.5, 0.5])
    ax[3].plot(xl, -xl + np.log10(F0.dwell_ms).median()
               + np.log10(F0.theta.clip(1e-4)).median(), "r--", lw=2,
               label="OU: slope −1")
    ax[3].set_xlim(xl); ax[3].set_xlabel("log10 θ"); ax[3].set_ylabel("log10 dwell")
    ax[3].set_title("pull does not predict dwell"); ax[3].legend(fontsize=8)
    for a in ax: a.grid(alpha=.3)
    save(fig, "fig_5_model.png")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser(
        description="segment: build the basin partition from raw data.  "
                    "validate: test it, from whatever segment produced.  "
                    "all: both.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--mode", choices=["segment", "validate", "all"], default="all",
                    help="segment = provenance, extract, faces, samples, runs, prep; "
                         "validate = statistics and figures from existing outputs")
    ap.add_argument("--stages", default=None,
                    help="explicit comma list, overrides --mode "
                         "(provenance,extract,faces,samples,runs,prep,validate,figures)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild outputs of the selected stages even if present")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    MODES = {"segment": "provenance,extract,faces,samples,runs,prep",
             "validate": "validate,figures",
             "all": "provenance,extract,faces,samples,runs,prep,validate,figures"}
    st = set(x.strip() for x in (a.stages or MODES[a.mode]).split(","))
    out = root/"output"/"reproduce"; out.mkdir(parents=True, exist_ok=True)
    for f in ["trajextract.py", "facelabel.py", "ou_visits.py"]:
        if not (root/f).exists(): raise SystemExit(f"missing {f} in {root}")
    ov = load_module(root/"ou_visits.py", "ou_visits")
    for k, v in [("KDE_PX", "kde_px"), ("GRID_PX", "grid_px"),
                 ("MIN_VISIT_MS", "min_visit_ms"), ("MAX_GAP_MS", "max_gap_ms"),
                 ("GUARD_MS", "guard_ms"), ("MIN_DENS", "min_dens")]:
        if float(getattr(ov, k)) != float(PARAMS[v]):
            raise SystemExit(f"ou_visits.{k}={getattr(ov, k)} but the document "
                             f"states {PARAMS[v]}")

    # ------------------------------------------------ segmentation pipeline
    if "provenance" in st: stage_provenance(root, out)
    if "extract" in st: stage_extract(root, a.force)
    if "faces" in st: stage_faces(root, a.force)
    # samples and prep also run (reuse-only) when validating, since the
    # statistics read their outputs; they rebuild only in segment mode
    need_data = bool(st & {"samples", "runs", "prep", "validate", "figures"})
    Bw = stage_samples(root, a.force and "samples" in st, ov) if need_data else None
    if "runs" in st: stage_runs(root, Bw, a.force)
    V = stage_prep(root, a.force and "prep" in st) if need_data else None

    # ------------------------------------------------------- validation
    if st & {"validate", "figures"}:
        V = derived(V)
        say(f"\n{len(V)} visits, {V.unit.nunique()} sessions, "
            f"{V.pid.nunique()} participants")
    if "validate" in st:
        R = Report()
        v_bandwidth(root, R, Bw)
        v_face_capture(root, R, V)
        v_effects(root, R, V)
        v_attendance(root, R, V)
        v_model(root, R, V, Bw)
        S = pd.DataFrame(R.rows); S.to_csv(out/"results.csv", index=False)
        pd.DataFrame(R.per_session, columns=["effect", "coef"]).to_csv(
            out/"per_session.csv", index=False)
        pd.DataFrame(R.nulls, columns=["test", "observed", "null"]).to_csv(
            out/"nulls.csv", index=False)
        with open(out/"results.txt", "w") as f:
            for sec, g in S.groupby("section", sort=False):
                f.write(f"\n§{sec}\n")
                for r in g.itertuples():
                    v = f"{r.value:.4g}" if isinstance(r.value, float) else str(r.value)
                    f.write(f"  {r.quantity:55} {v:>12}   {r.detail}\n")
        say(f"\nwrote {out}/results.txt and supporting tables")
    if "figures" in st: stage_figures(root, V, Bw)


if __name__ == "__main__":
    main()
