# Reproducibility Guide

This guide explains how to rebuild the environment, data, deployed model and application, how to rerun every experiment, and exactly what can be audited, reproduced or replayed. Every command was tested on 2026-10-27 in a fresh clone of commit `8d74f0e`, using a new virtual environment on macOS (Apple Silicon).

## 1. Three levels of reproducibility

| Level | Scope | How to use it |
|---|---|---|
| **Auditable tracked evidence** | `reports/` JSON results, `docs/images/` figures, and `reports/evidence/`: byte-for-byte copies of the Phase 3/4/6B metrics, Phase 6C summaries and fold metrics, and the canonical Phase 6D out-of-fold predictions, fold assignments, fold metrics and neighbour statistics | Inspect directly; no code needs to run |
| **Numerically reproducible experiments** | Phases 3, 4, 6B and 6C from a fresh clone (tested; section 6). Phases 6D and 6E in the original environment (each run twice; identical apart from timing fields). | Expect the same numbers. Files that record timings, run identifiers or the environment differ in bytes, and some files differ in the last floating-point digit. |
| **Exact replay** | Phases 6D and 6E, run unchanged | Needs the *original* checksum-protected inputs listed in `configs/phase6d.json` and `configs/phase6e.json`: the Phase 3/4 metrics, the Phase 6C outputs and, for 6E, the canonical Phase 6D run directory `results/phase6d/20261027T154454Z_d02c301c/`. These are Git-ignored and are not distributed. |

`reports/ARTIFACT_MANIFEST.json` lists every Git-ignored artifact with its size, SHA-256, MD5, generating command and any tracked copy.

## 2. Environment

| Requirement | Version used |
|---|---|
| Python | 3.12 (tested with 3.12.15) |
| Python packages | Pinned in `requirements.txt` (pandas 3.0.6, NumPy 2.5.3, RDKit 2026.3.6, scikit-learn 1.9.1, XGBoost 3.4.1, SciPy 1.18.1, Matplotlib 3.11.2, FastAPI 0.117.1, …) |
| OpenMP (macOS only) | `brew install libomp`; XGBoost fails to import without it |
| Node.js | `^20.19.0` or `>=22.12.0` (Vite 7 requirement; tested with Node 26.12 / npm 11.20) |

```bash
git clone https://github.com/Vdevannah/molecular-property-predictor.git
cd molecular-property-predictor
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check                      # expected: No broken requirements found.
python -c "import xgboost; print(xgboost.__version__)"   # on macOS this fails without libomp
```

## 3. Data (pinned and checksum-verified)

The ESOL file is downloaded from DeepChem commit `108af2176a5f876c2df1df922baadb02d647c826`, the only commit that has modified `datasets/delaney-processed.csv`. On 2026-10-27 the file at that commit was byte-identical to the file used throughout this project.

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/deepchem/deepchem/108af2176a5f876c2df1df922baadb02d647c826/datasets/delaney-processed.csv -o data/esol.csv
shasum -a 256 data/esol.csv
# expected: 8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189
python src/descriptors.py
md5 data/esol_descriptors.csv            # Linux: md5sum
# expected: 1a17c75935da0ab8961410c311558925
```

Stop if either checksum differs: every downstream result depends on these files.

## 4. Deployed model and application

```bash
python src/train_model.py
```

Expected output: Random Forest test RMSE 0.834945, R² 0.852515. In the fresh-clone test, `models/solubility_model.joblib` (MD5 `695cf080…`) and `models/feature_columns.joblib` (MD5 `17cb1fd0…`) were byte-identical to the originals.

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8003
curl http://127.0.0.1:8003/api/health    # {"status":"ok","model_loaded":true}

cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5175   # http://localhost:5175
```

The frontend reads `VITE_API_BASE_URL` (default `http://127.0.0.1:8003`). The backend allows CORS from ports 5175 on `localhost` and `127.0.0.1`. During `npm ci`, npm 11 may print an `install-scripts` warning; the tested build still succeeded.

## 5. Tests

```bash
python -m pytest -q
cd frontend && npm test && npm run build
```

| Environment | Backend result | Frontend |
|---|---|---|
| Original development environment (all artifacts present) | 145 passed | 12 passed, build OK |
| Fresh clone after sections 3–4 | **143 passed, 2 skipped** | 12 passed, build OK |

**The two fresh-clone skips**
- **Protected-artifact check:** `tests/test_scaffold_evaluation.py::test_protected_files_match_committed_checksums` checks the MD5 of the protected files recorded in `configs/phase6d.json`. Its outcome depends on the *historical experiment artifacts*: `results/phase4_metrics.json` and the six `results/phase6c/` outputs.
  - **None present** (a fresh clone, even after `train_model.py` has recreated the model and Phase 3 metrics) → **skipped**, with a message explaining that the original artifacts are not in the clone.
  - **Only some present** → **fails**, listing the missing files.
  - **All present** → every protected checksum is verified, including the model and Phase 3 metrics: **passes** in the original environment, **fails** on any mismatch.

  Rerunning Phases 4 and 6C on a fresh clone therefore makes this test fail by design. The Phase 3/4 metrics differ in the last floating-point digit and the Phase 6C files record timings, so a rerun is not the original evidence. This was confirmed on 2026-10-27.
- **Phase 6E refit check:** `tests/test_model_diagnostics.py::test_refit_gate_reproduces_phase6d_predictions_on_real_data` needs the canonical Phase 6D run directory.

## 6. Experiments

Run from the repository root after sections 3–4. Runtimes were measured on an 8-core Apple Silicon Mac.

| Phase | Command | Output | Fresh-clone result vs tracked evidence (tested 2026-10-27) | Runtime |
|---|---|---|---|---|
| 3 | `python src/train_model.py` | `results/phase3_metrics.json`, `models/` | Model files byte-identical; metrics equal apart from the last floating-point digit (RMSE …858 vs …859) | < 1 min |
| 4 | `python src/evaluate_model.py` | `results/phase4_metrics.json`, plots | Equal apart from the last floating-point digit of one R² | < 1 min |
| 6B | `python src/model_experiments.py` | `results/phase6b_holdout_metrics.json` | Byte-identical to `reports/evidence/phase6b/` | < 1 min |
| 6C | `python src/cross_validation.py` | `results/phase6c/` | `summary.csv` and `paired_comparisons.csv` byte-identical; `fold_metrics.csv` identical apart from `fit_seconds`; `config.json` differs only in `runtime_seconds` | about 3 min |
| 6D | `python src/scaffold_evaluation.py --benchmark`, then `python src/scaffold_evaluation.py` | `results/phase6d/<run_id>/`, `reports/phase6d/` | **Stops by design** with `Checksum mismatch` (section 7) | about 13 min |
| 6E | `python src/model_diagnostics.py --benchmark`, then `python src/model_diagnostics.py` | `results/phase6e/<run_id>/`, `reports/phase6e/`, `docs/images/phase6e/` | **Stops by design** (section 7) | about 2 min |

The last-digit differences in Phases 3 and 4 arise because those scripts predict with `n_jobs=-1`, where threaded summation of tree predictions can change the final floating-point bit.

**Warning:** Phases 6D and 6E overwrite their tracked reports. Running them rewrites `reports/phase6d/` or `reports/phase6e/` and `docs/images/phase6e/`. In the original environment, use `--reports-dir` (and, for 6E, `--figures-dir`) to write elsewhere when you only want to compare.

## 7. Why Phases 6D and 6E cannot run unchanged on a fresh clone

Both configurations record MD5 checksums of their inputs and stop when any differs. That is the point of the checks: they guarantee the analyses describe exactly the evaluated data and models.

- **Phase 6D** (`configs/phase6d.json`) protects the deployed model, the Phase 3/4 metrics and the Phase 6C outputs. On a fresh clone the Phase 3/4 metrics differ in the last floating-point digit, and the Phase 6C `fold_metrics.csv`, `config.json` and `benchmark.json` record timings. In the tested fresh clone, Phase 6D stopped with `Checksum mismatch for: results/phase3_metrics.json, results/phase4_metrics.json, results/phase6c/config.json, results/phase6c/fold_metrics.csv, results/phase6c/benchmark.json`.
- **Phase 6E** (`configs/phase6e.json`) protects 52 files and reads the canonical Phase 6D run directory `results/phase6d/20261027T154454Z_d02c301c/`. A new Phase 6D run would have a different directory name, and its `fold_metrics.csv`, `run_info.json` and `pip_freeze.txt` record timings or environment details.

**What a fresh clone *can* verify**
- The data, the descriptors and the deployed model, byte for byte.
- Every Phase 3–6C number, by rerunning and comparing with `reports/evidence/`.
- The Phase 6D and 6E results through the tracked evidence. `reports/evidence/phase6d/` contains the canonical out-of-fold predictions, fold assignments, fold metrics and neighbour statistics. Phase 6E's refit check reproduced those predictions to within 1.8 × 10⁻¹⁵. The Phase 6D protocol metrics in `reports/phase6d/` can be recomputed from these predictions. The Phase 6E tables also need refitted fold models, so they can be inspected but not recomputed from the evidence alone.

Exact Phase 6E replay elsewhere would need the original artifacts. Their checksums are in the manifest, but they are not distributed.

## 8. Verifying checksums

```bash
# Tracked evidence against the manifest (needs no untracked files):
python - <<'EOF'
import hashlib, json, pathlib
m = json.load(open("reports/ARTIFACT_MANIFEST.json"))
for f in m["files"]:
    copy = f["tracked_copy"]
    if copy:
        ok = hashlib.sha256(pathlib.Path(copy).read_bytes()).hexdigest() == f["sha256"]
        print("OK " if ok else "BAD", copy)
EOF
```

The manifest's `verify` field holds a one-line check of any local artifacts against their recorded SHA-256 values. Files that are absent are not reported as mismatches.
