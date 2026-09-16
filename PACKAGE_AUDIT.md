# Package audit

- Controlled individual-level CHARLS data are not included.
- Aggregate table and figure data are in `source_data/`.
- `charls_bidirectional_covariate_models.py` and `figure_helpers.py` are included because the five-wave scripts import them.
- The primary models use `rural_nbs`, defined as rural versus urban community classification.
- Strict 11-item completeness is assessed from raw BADL/IADL item responses before structural-skip recoding; the corrected sensitivity sample includes 17,682 people and 42,468 intervals.
- In the 2013 Exit Interview-only sensitivity analysis, 31 deaths identified only from non-Exit-Interview sources are censored as unknown vital/state outcomes rather than recoded as known alive; the resulting sample includes 21,206 people and 62,106 intervals.
- The two 126-parameter primary models and all four sensitivity analyses were rerun after this correction; all optimisations converged.
- Analysis scripts should be rerun by authorised CHARLS users after placing data as described in `DATA_LAYOUT.md`.
- Licence and citation terms are governed by the corresponding public repository and archived release.
