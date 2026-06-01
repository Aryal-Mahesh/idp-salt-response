# Sequence charge decoration organizes salt-response regimes of intrinsically disordered proteins

Code and data for an interpretable machine-learning study of how intrinsically
disordered protein (IDP) conformational ensembles respond to monovalent salt,
built on a 511-sequence library and 2,555 CALVADOS-2 coarse-grained simulations.

The central result: the length-weighted sequence charge decoration, **SCD x N**,
is the dominant coordinate organizing salt response,it accounts for roughly
40% of total SHAP feature attribution and exceeds the next feature threefold,
with a sigmoidal dependence whose sign sets the *direction* of response and whose
magnitude sets the *strength*.

This repository accompanies the manuscript (preprint link to be added).

---

## What's here

```
.
├── input/                       # CALVADOS-2 residue parameters + input fasta
│   ├── residues_CALVADOS2.csv
│   ├── idr.fasta
│   └── library.fasta
│
├── library/                     # Methods section 1 -- sequence library construction
│   ├── generate_subset1_kappa.py        # 195 kappa-variants (Das-Pappu, K/E on G/S/A/P)
│   ├── generate_subset2_NCPR.py         # 108 NCPR series (fixed FCR = 0.40)
│   ├── generate_subsets34_idrome.py     # 150 IDRome-stratified + 58 low-FCR naturals
│   ├── merge_library.py                 # assembles the four subsets -> library.csv
│   ├── fix_hydropathy.py                # uniform Kyte-Doolittle mean hydropathy
│   ├── plot_library_coverage.py         # library coverage figures
│   ├── subset1_kappa_variants.csv       # per-subset outputs
│   ├── subset2_ncpr_series.csv
│   ├── subset3_idrome_stratified.csv
│   ├── subset4_idrome_lowfcr.csv
│   ├── library.csv                      # the canonical 511-sequence library
│   └── plots/
│
└── sweep/                       # Methods sections 2-5 -- simulations, analysis, ML
    ├── build_fasta.py, build_jobs.py    # enumerate the 2,555-job manifest (jobs.csv)
    ├── prepare_one.py, prepare_all.py   # build per-simulation inputs
    ├── submit_full.pbs                  # cluster submission (PBS array)
    ├── collect_observables.py           # trajectories -> observables_{long,wide}.csv
    ├── build_master_table.py            # -> master_table.csv (canonical analysis table)
    ├── fit_salt_response.py             # per-sequence dRg/d[salt] fits
    ├── classify_regimes.py              # four-regime assignment
    ├── physics_baseline.py              # feature definitions + ridge baseline
    ├── ml_xgboost.py                    # ridge vs XGBoost regression, random + LOSO CV
    ├── regime_classifier.py             # logistic + XGBoost regime classifier
    ├── shap_analysis.py                 # SHAP feature attribution
    ├── make_fig4_shap.py                # combined SHAP figure
    ├── plot_regime_landscape.py         # regime maps (Fig 2 and others)
    ├── plot_pred_vs_actual.py           # predicted-vs-actual figure (Fig 3)
    ├── master_table.csv                 # 511 sequences x derived columns (frozen)
    ├── ml_predictions.csv               # master table + model predictions
    ├── shap_feature_importance.csv      # mean |SHAP| per feature
    └── plots/                           # all figures
```

Note: the figure scripts read their inputs (`master_table.csv`,
`ml_predictions.csv`) by relative path and write to `plots/`, so run them from
inside `sweep/`. `physics_baseline.py` defines the eight features via
`build_features()` and is imported by every other ML script -- it is the single
source of truth for the feature set.

---

## The library (511 sequences)

| Subset | n | Source |
|---|---|---|
| kappa-variants | 195 | synthetic; K/E on a Gly/Ser/Ala/Pro background, charge patterning varied at fixed composition |
| NCPR series | 108 | synthetic; net charge titrated -0.4 to +0.4 at fixed FCR = 0.40 |
| IDRome stratified | 150 | natural; stratified over (SCD, length) from IDRome |
| IDRome low-FCR | 58 | natural; FCR < 0.10, the biologically rare weakly-charged regime |

Natural sequences are drawn from IDRome (Tesei et al. 2024), restricted to
entries flagged as confidently disordered (is_idp = True) and lengths 30-300.
Synthetic descriptors (FCR, NCPR, kappa) are computed with localCIDER; natural
descriptors are taken from IDRome's published values. SCD is computed uniformly
via the Sawle-Ghosh definition across all subsets.

---

## Reproducing the pipeline

Run with the `idp-salt` environment active (see `environment.yml`). Each stage
consumes the previous stage's output.

1. **Build the library** (`library/`). Regenerating the IDRome subsets requires
   the IDRome database (see *Data availability*); the synthetic subsets
   regenerate from the scripts alone. `merge_library.py` produces `library.csv`.

2. **Run simulations** (`sweep/`). `build_jobs.py` enumerates the 2,555 jobs
   (511 sequences x five salts: 50, 100, 150, 300, 500 mM). `prepare_all.py`
   builds per-job inputs; `submit_full.pbs` submits the array (edit the
   allocation and module lines for your cluster). All five salt points,
   including 150 mM, are simulated in-house under one protocol.
   `collect_observables.py` reduces trajectories to `master_table.csv`.

3. **Fit and classify** (`sweep/`). `fit_salt_response.py` computes the slope
   dRg/d[salt]; `classify_regimes.py` assigns one of four regimes
   (salt-insensitive if |relative change| < 3%, otherwise polyelectrolyte
   contraction / polyampholyte swelling / non-monotonic by the shape of the
   discrete Rg-vs-salt curve).

4. **Train and evaluate** (`sweep/`). `ml_xgboost.py` runs the ridge baseline and
   gradient-boosted regressor under random 5-fold and leave-one-subset-out CV;
   `regime_classifier.py` runs the four-class classifier.

5. **Interpret** (`sweep/`). `shap_analysis.py` and `make_fig4_shap.py` produce
   the SHAP attribution and figures.

All ML scripts use a fixed random seed (42).

---

## Key results

- **SCD x N is the master predictor** (mean |SHAP| = 0.643 nm/M; next feature
  SCD alone, 0.290), with a sigmoidal dependence saturating near +2.2 nm/M for
  strong polyampholytes and -1.8 nm/M for strong polyelectrolytes.
- **Regression** (predicting dRg/d[salt]): ridge R2 = 0.83 / XGBoost R2 = 0.97
  under random 5-fold CV; ridge R2 = -0.13 / XGBoost R2 = 0.60 under
  leave-one-subset-out. The LOSO value is the honest generalization estimate;
  random-CV is optimistic because compositionally related sequences leak across
  folds.
- **Regime classification**: 0.726 accuracy (logistic baseline 0.693); per-class
  F1 of 0.87 (PA swelling), 0.81 (PE contraction), 0.69 (salt-insensitive),
  0.46 (non-monotonic), with zero confusion between the two directional regimes.

---

## Limitations (read before reusing)

- CALVADOS-2 represents salt only through Debye-Huckel screening; it does not
  capture residue-specific salting-out or Hofmeister effects. Predictions are
  CALVADOS-2-derived and most limited for weakly charged sequences.
- The random-CV R2 = 0.97 is interpolation-inflated; treat R2 = 0.60 (LOSO) as
  the deployment-relevant number for unseen sequence families.
- For the salt-insensitive regime, R2 is uninformative (near-zero target
  variance); use MAE (0.11 nm/M there).
- The study is simulation-based; no direct experimental validation is included.

---

## Data availability

- `library/library.csv` and `sweep/master_table.csv` contain the sequences,
  descriptors, and all reduced observables needed to reproduce every figure and
  number in the paper.
- The raw simulation trajectories (~9 GB) are too large for this repository and
  are archived at Zenodo: **[DOI to be added]**.
- The IDRome database used to build the natural subsets is from Tesei et al.
  (2024), Zenodo 10.5281/zenodo.10251736.

---

## Dependencies

CALVADOS-2 (Tesei & Lindorff-Larsen) is required to run the simulations; install
it from the authors' distribution and pin the version in `environment.yml`. Other
key dependencies: Python 3.11, OpenMM, localCIDER, xgboost, shap, scikit-learn,
pandas, numpy, matplotlib.

---

## Citation

Citation details and the preprint link will be added on posting. If you use this
code or data, please cite the accompanying manuscript along with the IDRome and
CALVADOS-2 sources.
