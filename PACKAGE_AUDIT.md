# Package audit

- Controlled individual-level CHARLS data are not included.
- Aggregate table and figure data are in `source_data/`.
- `charls_bidirectional_covariate_models.py` and `generate_scirep_figures.py` are included because the five-wave scripts import them.
- The primary models use `rural_nbs`, defined as rural versus urban community classification.
- Strict 11-item completeness is assessed from raw BADL/IADL item responses before structural-skip recoding; the corrected sensitivity sample includes 17,682 people and 42,468 intervals.
- Analysis scripts should be rerun by authorised CHARLS users after placing data as described in `DATA_LAYOUT.md`.
- A public repository licence and archived DOI remain author decisions before public release.
