# OU basin segmentation of gaze — code

Segments eye-tracking trajectories into data-driven regions (basins) and validates them against independently labelled faces. The method and results are in `ou_gaze.pdf`; this file covers running and changing the code.

## Contents

| file | role |
|---|---|
| `reproduce.py` | driver: runs the pipeline and the validation, records provenance |
| `trajextract.py` | stage 1 — parses EyeLink `.asc` into samples, saccades, trials |
| `facelabel.py` | stage 2 — face ellipses from 68-point landmarks |
| `ou_visits.py` | stage 3 — KDE, basins, Λ_k, membership, visits, for one bandwidth |
| `OU_field.py` | per-image field figures; diagnostic only, not run by `reproduce.py` |

## Requirements

Python 3.10+ with `numpy pandas scipy statsmodels matplotlib pyarrow` (pyarrow optional; samples fall back to CSV). The face stage also needs `torch` and `face_alignment`; a GPU is used if available, otherwise CPU (slow, ~1–3 s per image).

On WSL/conda, if `import face_alignment` fails with `CXXABI_1.3.15 not found`, point the loader at conda's libstdc++:

```bash
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib -n <env>
conda deactivate && conda activate <env>
```

## Data layout

Run from a directory containing the scripts and:

```
eyedata/sub-<pid>/ses-<ss>/eye/*.asc
ImageTrials/IAPS/*.jpg
ImageTrials/ABMimages/<pid>_trials4experiment/*
```

Participant and session are taken from the directory names, never the `.asc` filename (those are inconsistent; one file under `sub-D216` is named for D543).

## Quick start

```bash
python3 reproduce.py --root . --mode all        # everything, from raw data
python3 reproduce.py --root . --mode segment    # build the partition only
python3 reproduce.py --root . --mode validate   # statistics + figures from existing outputs
python3 reproduce.py --root . --stages figures  # redraw figures only
```

`--mode validate` never rebuilds the segmentation; it only writes to `output/reproduce/`.

## Stages

| stage | does | writes | cost |
|---|---|---|---|
| `provenance` | SHA-256 of every input and script, versions, parameters, seeds | `output/reproduce/provenance.json`, `inputs.csv` | slow first time (hashes cached) |
| `extract` | `trajextract.py` per recording | `output/extracted/ses-<ss>/` | minutes per file |
| `faces` | `facelabel.py` per stimulus folder | `output/faces/<set>/face_bounds.csv` | GPU or slow CPU |
| `samples` | per-image Silverman ĥ_im, robust Silverman, increment kurtosis | `output/bandwidth_table.csv` | one pass over samples |
| `runs` | `ou_visits.py` × 3 bandwidths × every session | `output/{h26,hsilv,hmean}/<pid>_ses-<ss>/visits.csv` | ~1 h on CPU |
| `prep` | face identity, depth q, condition, stimulus timing onto h26 visits | `output/visits_full.csv` | seconds |
| `validate` | every statistic in the PDF | `output/reproduce/results.txt` and tables | minutes |
| `figures` | one figure per PDF section | `output/reproduce/fig_*.png` | seconds |

`--mode segment` = provenance → prep. `--mode validate` = validate, figures (samples and prep run in reuse-only form to load their outputs). `--stages a,b,c` overrides `--mode`.

### Caching and rebuilding

Each stage skips work whose output already exists. `--force` rebuilds the outputs of the selected stages only. Specific cases:

- a `visits.csv` without the Λ_k eigenvalue columns (`th1`, `th2`) is regenerated automatically;
- an older `bandwidth_table.csv` missing the shape columns is completed from `output/validation_ok/kde_shape.csv` if present, never recomputed in validate mode;
- an older `visits_full.csv` missing stimulus timing has it joined in memory.

To rebuild one session, delete its `visits.csv` in each condition folder and rerun `--stages runs` without `--force`.

## Outputs

```
output/
  extracted/ses-<ss>/<pid>_ses-<ss>_{samples,saccades,fixations,trials}.*
  faces/<set>/face_bounds.csv            one ellipse per detected face
  bandwidth_table.csv                    per participant × image
  h26/ hsilv/ hmean/<pid>_ses-<ss>/visits.csv
  visits_full.csv                        h26 visits + face_id, face_q, condition
  reproduce/
    results.txt                          every number, in document order
    results.csv                          same, machine-readable
    per_session.csv                      per-session coefficients of each dwell effect
    nulls.csv                            permutation null draws
    face_capture.csv                     O_i, U_i, A_iU per image-session
    capture_abm_sessions.csv             O/A_U per ABM session
    capture_iaps_sessions.csv            O_j/A_Cj per IAPS session
    reentry_table.csv                    brief share and sustained dwell by re-entry
    attendance_by_area.csv               attended share by face-area decile
    fig_4_1_bandwidth.png  fig_4_2_face_capture.png
    fig_5_effects.png  fig_5_reentry.png  fig_5_model.png
    provenance.json  inputs.csv
```

Key `visits.csv` columns: `basin`, `mu_x`, `mu_y` (μ_k), `th1`, `th2` (eigenvalues of Λ_k), `theta` (their mean), `v11…v22` (eigenvectors), `aniso`, `basin_px`, `on_face`, `entry_ms`, `exit_ms`, `dwell_ms`, `re` (re-entry index), `sigma2`, `h_px`. `work`, `f_mean` are written but not interpreted.

## Changing things

Parameters live in two places. **Estimation** constants are at the top of `ou_visits.py`; **validation** and bookkeeping constants are the `PARAMS` dict at the top of `reproduce.py`. `reproduce.py` stops if these disagree for `KDE_PX`, `GRID_PX`, `MIN_VISIT_MS`, `MAX_GAP_MS`, `GUARD_MS`, `MIN_DENS`, so change both together.

| to change | edit | then rerun |
|---|---|---|
| bandwidth h | `KDE_PX` in `ou_visits.py`, `kde_px` in `PARAMS`, the `h26` entry in `CONDS` | `--stages runs,prep,validate,figures --force` |
| grid spacing h_g | `GRID_PX` (+ `grid_px`) | runs onward |
| density floor (basin condition 1) | `np.nanpercentile(P, 60)` in `maxima()` in `ou_visits.py` — hard-coded, not a constant | runs onward |
| cleaning (blink guard, saccade margin, velocity cap) | `GUARD_MS`, the `±4` in `clean_mask()`, `VEL_MAX_PX` | samples, runs onward |
| minimum samples per trial / image | the `20` and `200` in `ou_visits.py` `main()`, and `min_samples_*` in `PARAMS` | samples, runs onward |
| visit rules | `MIN_VISIT_MS`, `MAX_GAP_MS` | runs onward |
| membership | `--membership nearest` labels every sample (no unlabelled state); add it to the flags in `CONDS` | runs onward |
| face ellipse size | `ELLIPSE` (1.06) in `facelabel.py` | faces, runs onward |
| practice trials | `N_PRACTICE` in `trajextract.py` | everything |
| truncation, brief-visit threshold, floor radius, matching radius, ARI grid | `truncation_ms`, `brief_ms`, `floor_px`, `match_px`, `ari_grid_ms` in `PARAMS` | validate |
| null draws, permutations, bootstrap | `n_uniform`, `n_alt`, `n_perm`, `n_boot` in `PARAMS` | validate |
| random seeds | `SEEDS` in `reproduce.py` | validate |

"runs onward" = `--stages runs,prep,validate,figures --force`; add `samples` when the change affects which samples are retained.

### Adding a bandwidth condition

Add an entry to `CONDS` in `reproduce.py`, e.g.

```python
CONDS["h40"] = ["--kde-mode", "fixed", "--kde-px", "40"]
```

`--stages runs` then builds `output/h40/`. The bandwidth comparisons in `v_bandwidth()` list the three condition names explicitly; add the new one to the `Vs` dict and the pairs if it should enter the Spearman and correspondence tables.

### Restricting saccades to one eye

Binocular recordings (the D cohort) currently contribute both eyes' saccades, which lowers σ² for that cohort. D-invariant results (basins, membership, visits, dwell) are unaffected, and the document's numbers depend on the current behaviour. To keep only the left eye, capture the eye label in `trajextract.py`:

```python
RE_EFIX  = re.compile(r"^EFIX\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)")
RE_ESACC = re.compile(r"^ESACC\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                      r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
```

then skip matches whose first group is not `L` and shift the remaining group indices by one. Rerun everything with `--force`.

### Running only part of the validation

The calls in `main()` under `if "validate" in st:` are independent: `v_bandwidth`, `v_face_capture`, `v_effects`, `v_attendance`, `v_model`. Comment out any you don't need. The figure stage needs every table the five functions write, so run it only after a full validate.

## Running the stages by hand

```bash
python3 trajextract.py --root . --asc eyedata/sub-T365/ses-05/eye/<file>.asc \
                       --outdir output/extracted/ses-05
python3 facelabel.py --imgdir ImageTrials/IAPS --outdir output/faces/IAPS [--no-overlay]
python3 ou_visits.py --root . --pid T365 --ses 05 --outdir output/h26/T365_ses-05 \
                     --faces output/faces/IAPS --faces output/faces/T365 \
                     --kde-mode fixed --kde-px 26          # or --kde-mode silverman
python3 OU_field.py --root . --pid T365 --ses 05 --kde-px 26 \
                    --faces output/faces/IAPS --imgdir ImageTrials/IAPS \
                    --outdir output/ou_field/T365_ses-05 --max-figs 20
```

`facelabel.py --images 2208,7515` restricts to named stems and writes overlays for visual checking unless `--no-overlay`. `ou_visits.py --images` does the same for estimation.

## Reproducibility

`provenance.json` records script hashes, library versions, `PARAMS` and `SEEDS`; `inputs.csv` lists every input file with its size and SHA-256. With the same inputs, scripts and seeds, `results.txt` is identical across runs. Input hashes are cached in `output/reproduce/hash_cache.json` by path, size and modification time.

## Notes

- Faces are detected in every stimulus folder, so IAPS images appearing inside ABM folders are detected more than once; duplicates are dropped by `(stem, face_id)`.
- Neutral images yield fewer face detections, which confounds the third valence condition; valence tests use Negative vs Positive only.
- `OU_field.py` estimates σ² slightly differently from `ou_visits.py` and colours paths by a local rate that is not the θ of the document; use it for inspection only.

## Use of AI (Claude Opus 5.0 on High) 

An AI assistant (Claude, Anthropic) was used during development and write-up. What it did, and what it did not:

**Human-authored**
- The model formulation in `ou_gaze.pdf` §1–2 and the validation design in §4.1–4.2 (text edited by the AI only for spelling and LaTeX formatting).
- The pipeline design, validation procedure, and choice of what to test.
- Review and acceptance of every change and every reported number.

**AI-assisted code**
- `reproduce.py` was written by the AI: the pipeline driver, provenance record, every validation statistic, and the figures.
- Changes to the stage scripts: repointing paths to the current directory layout (`--root`, `--ext`), taking participant and session from the directory in `trajextract.py`, the `--kde-px` / `--kde-mode` options and per-image Silverman bandwidth in `ou_visits.py`, and writing the Λ_k eigenvalues and eigenvectors to `visits.csv`.
- The docstrings and comments of all four stage scripts were rewritten by the AI to match the document's notation; the code itself was checked unchanged against the prior versions (identical syntax trees apart from docstrings).
- Exploratory analysis scripts used during development (bandwidth comparisons, face-capture nulls, per-session effects), whose final forms are folded into `reproduce.py`.

**AI-assisted analysis and writing**
- Choice and design of several tests: the permutation nulls for detailed balance and between-face differences, the uniform and same-participant nulls for face capture, per-session (two-stage) estimation, and the re-entry split into brief and sustained visits.
- Interpretation of intermediate results, including corrections of its own earlier misreadings (the transition asymmetry, first read as a finding, is counting bias).
- Drafting of `ou_gaze.pdf` §3 (pipeline and fitting procedure), §4.3 (methods for the result tests) and §5 (results), and of this README.

Every number in §5 is produced by `reproduce.py` from the raw data and can be regenerated with `--mode all`.
