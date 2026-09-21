#!/usr/bin/env python3
"""
trajextract.py — stage 1 of the segmentation pipeline: extraction.

ROLE IN THE DOCUMENT
  Produces the samples S_k = (x_k, y_k) of §2, the trial structure that
  defines the stimulus period, and the EyeLink saccades that the cleaning of
  §3 removes.  Nothing is estimated here; the file is only parsed.

WHAT THE ASC CONTAINS (verified against T271/ses-05)

  MSG t TRIALID n                 trial begins
  START t LEFT SAMPLES EVENTS     recording on
  MSG t -6 DISPLAY_CROSSHAIR      crosshair drawn
  MSG t 0  TIMER_CROSSHAIR        crosshair timer expired
  MSG t -14 DISPLAY_STIMULI       image drawn
  MSG t 0  TIMER_STIMULI          stimulus timer expired
  MSG t -14 DISPLAY_VALENCE_Q     rating screens follow
  MSG t 0  MOUSE_RESP_<n>_VA      response
  MSG t CLEAR_TTL_END_TRIAL
  END t SAMPLES EVENTS RES rx ry  recording off; rx, ry = px per degree
  MSG t !V TRIAL_VAR image_file … the trial's variables, logged AFTER END
  MSG t TRIAL_RESULT 0

  The integer after the timestamp on DISPLAY_* lines is the display delay
  reported by the presentation software (negative: the message preceded the
  retrace).  Onset is t + delay.

  TRIAL_VAR appears after END, so trial variables — image and condition —
  are attached to the trial that has just finished.  Assigning them forward
  would shift every image label by one.

STIMULUS PERIOD
  TIMER_CROSSHAIR to TIMER_STIMULI, 6 s.  Every sample carries a phase;
  only phase == "stimulus" enters the KDE.  The crosshair duration is
  jittered per trial and recorded but not used in the document.

EVENTS
  EFIX  <eye> start end dur x y pupil
  ESACC <eye> start end dur sx sy ex ey amp_deg peak_vel
  EyeLink's own detection.  The eye label is not filtered: a binocular
  recording (the D cohort) contributes both eyes' saccades, while the sample
  columns read are the left eye's.  Both eyes' saccade intervals are then
  removed by the cleaning of §3, which lowers σ² for that cohort (§5, the
  84 vs 304 px²/s difference).  Basin geometry, membership, visits and dwell
  are D-invariant and unaffected.  Kept as is so the document's numbers
  reproduce.

PRACTICE TRIALS
  The first two trials are practice; they are kept, flagged, and excluded
  downstream.

PARTICIPANT AND SESSION
  Taken from the directory eyedata/sub-<pid>/ses-<ss>/eye/, never the
  filename, which is inconsistent across recordings.

OUTPUT (--outdir, default output/extracted/ses-<ss>)
  <pid>_ses-<ss>_trials.csv      one row per trial: onsets, image, condition
  <pid>_ses-<ss>_fixations.csv   EyeLink fixations, with trial and phase
  <pid>_ses-<ss>_saccades.csv    EyeLink saccades, with trial and phase
  <pid>_ses-<ss>_samples.*       gaze samples, trial- and phase-indexed
  <pid>_ses-<ss>_asc_meta.json   counts, resolution, sample rate
"""
from __future__ import annotations
import argparse, glob, json, re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

N_PRACTICE = 2

RE_MSG   = re.compile(r"^MSG\s+(\d+)\s+(.*)$")
RE_DELAY = re.compile(r"^(-?\d+)\s+(.*)$")
RE_VAR   = re.compile(r"^!V\s+TRIAL_VAR\s+(\S+)\s*(.*)$")
RE_START = re.compile(r"^START\s+(\d+)")
RE_END   = re.compile(r"^END\s+(\d+).*?RES\s+([\d.]+)\s+([\d.]+)")
RE_EFIX  = re.compile(r"^EFIX\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)")
RE_ESACC = re.compile(r"^ESACC\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                      r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"
                      r"(\S+)\s+(\S+)")
RE_SAMP  = re.compile(r"^(\d+)\s+(\S+)\s+(\S+)\s+(\S+)")


def _num(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan


def parse(path: str) -> dict:
    trials, fixes, saccs, samples = [], [], [], []
    cur = None
    res_x = res_y = np.nan
    warnings = []

    for line in open(path, errors="replace"):
        line = line.rstrip("\n")

        m = RE_SAMP.match(line)
        if m:
            t = int(m.group(1))
            def _f(v):
                try: return float(v)
                except ValueError: return np.nan
            x, y, p = _f(m.group(2)), _f(m.group(3)), _f(m.group(4))
            samples.append((t, x, y, p))
            continue

        m = RE_EFIX.match(line)
        if m:
            fixes.append(dict(start_ms=int(m.group(1)), end_ms=int(m.group(2)),
                              dur_ms=int(m.group(3)), x=_num(m.group(4)),
                              y=_num(m.group(5)), pupil=_num(m.group(6))))
            continue

        m = RE_ESACC.match(line)
        if m:
            saccs.append(dict(start_ms=int(m.group(1)), end_ms=int(m.group(2)),
                              dur_ms=int(m.group(3)),
                              sx=_num(m.group(4)), sy=_num(m.group(5)),
                              ex=_num(m.group(6)), ey=_num(m.group(7)),
                              amp_deg=_num(m.group(8)),
                              peak_vel=_num(m.group(9))))
            continue

        m = RE_START.match(line)
        if m and cur is not None:
            cur["rec_start_ms"] = int(m.group(1)); continue

        m = RE_END.match(line)
        if m:
            if cur is not None: cur["rec_end_ms"] = int(m.group(1))
            res_x, res_y = _num(m.group(2)), _num(m.group(3))
            continue

        m = RE_MSG.match(line)
        if not m:
            continue
        t, body = int(m.group(1)), m.group(2).strip()

        # DISPLAY_* lines carry the display delay before the label;
        # onset = t + delay
        delay, label = 0, body
        d = RE_DELAY.match(body)
        if d:
            delay, label = int(d.group(1)), d.group(2).strip()

        if label.startswith("TRIALID"):
            if cur is not None: trials.append(cur)
            cur = dict(trialid=int(label.split()[1]), trialid_ms=t)
            continue
        if cur is None:
            continue

        if label == "DISPLAY_CROSSHAIR":
            cur["cross_on_ms"] = t + delay
        elif label == "TIMER_CROSSHAIR":
            cur["cross_off_ms"] = t + delay
        elif label == "DISPLAY_STIMULI":
            cur["stim_on_ms"] = t + delay
        elif label == "TIMER_STIMULI":
            cur["stim_off_ms"] = t + delay
        elif label.startswith("DISPLAY_VALENCE_Q"):
            cur.setdefault("valence_on_ms", t + delay)
        elif label.startswith("MOUSE_RESP"):
            parts = label.split("_")
            cur.setdefault(f"resp_{parts[-1].lower()}_ms", t)
        elif label == "CLEAR_TTL_END_TRIAL":
            cur["trial_end_ms"] = t
        else:
            v = RE_VAR.match(label)
            if v:
                cur[f"var_{v.group(1).strip('_').lower()}"] = v.group(2).strip()
    if cur is not None:
        trials.append(cur)

    T = pd.DataFrame(trials)
    if T.empty:
        raise ValueError(f"no TRIALID found in {path}")
    T = T.sort_values("trialid_ms").reset_index(drop=True)
    T["trial_i"] = np.arange(len(T))
    T["practice"] = T.trial_i < N_PRACTICE
    for a, b, name in (("cross_on_ms", "cross_off_ms", "cross_dur_ms"),
                       ("stim_on_ms", "stim_off_ms", "stim_dur_ms")):
        if a in T and b in T: T[name] = T[b] - T[a]

    S = pd.DataFrame(samples, columns=["t_ms", "x", "y", "pupil"])
    F = pd.DataFrame(fixes)
    C = pd.DataFrame(saccs)

    # every sample and event gets its trial and its phase; the KDE of §2
    # uses only phase == 'stimulus' of non-practice trials
    def phase_of(ts, tr):
        i = np.searchsorted(tr.trialid_ms.values, ts, side="right") - 1
        i = np.clip(i, 0, len(tr) - 1)
        out = pd.DataFrame({"trial_i": tr.trial_i.values[i],
                            "trialid": tr.trialid.values[i]})
        co = tr.cross_on_ms.values[i]; cf = tr.cross_off_ms.values[i]
        so = tr.stim_on_ms.values[i];  sf = tr.stim_off_ms.values[i]
        ph = np.full(len(ts), "other", dtype=object)
        ph[(ts >= co) & (ts < cf)] = "crosshair"
        ph[(ts >= so) & (ts < sf)] = "stimulus"
        ph[ts >= sf] = "rating"
        out["phase"] = ph
        out["t_rel_stim_ms"] = ts - so
        out["image"] = (tr["var_image_file"].values[i]
                        if "var_image_file" in tr else "")
        out["condition"] = (tr["var_condition"].values[i]
                            if "var_condition" in tr else "")
        out["practice"] = tr.practice.values[i]
        return out

    for D, key in ((S, "t_ms"), (F, "start_ms"), (C, "start_ms")):
        if not D.empty:
            D[["trial_i", "trialid", "phase", "t_rel_stim_ms",
               "image", "condition", "practice"]] = \
                phase_of(D[key].values, T).values

    meta = dict(
        source=str(Path(path).resolve()),
        parsed=datetime.now().isoformat(timespec="seconds"),
        n_trials=len(T), n_practice=int(T.practice.sum()),
        n_samples=len(S), n_fixations=len(F), n_saccades=len(C),
        res_px_per_deg=[res_x, res_y],
        sample_rate_hz=float(1000.0 / np.median(np.diff(S.t_ms.values[:5000])))
                       if len(S) > 5000 else np.nan,
        missing_sample_frac=float(S.x.isna().mean()) if len(S) else np.nan,
        t_first_ms=int(S.t_ms.min()) if len(S) else None,
        t_last_ms=int(S.t_ms.max()) if len(S) else None,
        trial_vars=sorted(c[4:] for c in T.columns if c.startswith("var_")),
        warnings=warnings,
    )
    return dict(trials=T, fixations=F, saccades=C, samples=S, meta=meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asc", help="single .asc file")
    ap.add_argument("--root", default=".",
                    help="test root holding eyedata/sub-*/ses-*/eye/*.asc")
    ap.add_argument("--nas", default=None, help="(legacy, unused)")
    ap.add_argument("--ses", default=None, help="e.g. 05; omit for all")
    ap.add_argument("--pid", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-samples", action="store_true",
                    help="skip the sample table (it is by far the largest)")
    a = ap.parse_args()

    root = Path(a.root)
    files = ([a.asc] if a.asc else
             sorted(glob.glob(str(root/"eyedata"/"sub-*"/"ses-*"/"eye"/"*.asc"))))
    if a.ses:
        files = [f for f in files if f"ses-{a.ses}" in f]
    if a.pid:
        files = [f for f in files if f"sub-{a.pid}" in f or a.pid in Path(f).parts[-4]]
    if not files:
        raise FileNotFoundError(f"no .asc under {root}/eyedata/sub-*/ses-*/eye/")

    for f in files:
        nm = Path(f).name
        # participant and session come from the directory, never the
        # filename: the .asc names are inconsistent (D337s17.asc,
        # T365_ses-05_..., and one file under sub-D216 named for D543)
        parts = Path(f).resolve().parts
        pid = parts[-4].replace("sub-", "")
        ses = parts[-3].replace("ses-", "")
        if pid not in nm:
            print(f"  NOTE filename/dir mismatch: {nm} under sub-{pid}")
        out = (Path(a.outdir) if a.outdir else
               root/"output"/"extracted"/f"ses-{ses}")
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{pid}_ses-{ses}"
        print(f"\n=== {nm} -> {stem} ===", flush=True)
        try:
            R = parse(f)
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {e}"); continue
        M = R["meta"]
        print(f"  trials {M['n_trials']} ({M['n_practice']} practice)  "
              f"samples {M['n_samples']}  fix {M['n_fixations']}  "
              f"sacc {M['n_saccades']}")
        print(f"  {M['sample_rate_hz']:.0f} Hz  res {M['res_px_per_deg'][0]:.1f} px/deg  "
              f"missing {100*M['missing_sample_frac']:.1f}%")
        T = R["trials"]
        if "cross_dur_ms" in T:
            cd = T.cross_dur_ms.dropna()
            print(f"  crosshair {cd.median():.0f} ms "
                  f"[{cd.min():.0f}-{cd.max():.0f}]  jitter sd {cd.std():.0f}")
        if "stim_dur_ms" in T:
            sd = T.stim_dur_ms.dropna()
            print(f"  stimulus  {sd.median():.0f} ms "
                  f"[{sd.min():.0f}-{sd.max():.0f}]")
        if "var_condition" in T:
            print("  conditions: " + ", ".join(
                f"{k}={v}" for k, v in T.var_condition.value_counts().items()))
        T.to_csv(out/f"{stem}_trials.csv", index=False)
        R["fixations"].to_csv(out/f"{stem}_fixations.csv", index=False)
        R["saccades"].to_csv(out/f"{stem}_saccades.csv", index=False)
        if not a.no_samples:
            S = R["samples"]
            try:
                S.to_parquet(out/f"{stem}_samples.parquet", index=False)
                sp = f"{stem}_samples.parquet"
            except Exception:
                S.to_csv(out/f"{stem}_samples.csv", index=False)
                sp = f"{stem}_samples.csv"
            print(f"  wrote {sp}")
        (out/f"{stem}_asc_meta.json").write_text(json.dumps(M, indent=2))
        print(f"  wrote {stem}_trials.csv / _fixations.csv / _saccades.csv")


if __name__ == "__main__":
    main()
