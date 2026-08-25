# Five-wave CHARLS pain-function multistate analysis

Reproducible analysis code for the study **Bidirectional associations between musculoskeletal pain burden and functional limitations in middle-aged and older Chinese adults: a multistate analysis**.

## What is included

- harmonisation and quality control for 2011, 2013, 2015, 2018, and 2020;
- wave-specific construction and audit of the 2013 sample-information file required by the public-release layout;
- analytic cohort and adjacent-wave interval construction;
- continuous-time multistate models for pain and function, household-cluster robust inference, model-structure tests, and sensitivity analyses;
- publication figures and Scientific Reports document-generation code;
- non-disclosive aggregate table/figure source data in `source_data/` and numerical starting values.

## Data availability

CHARLS participant data are controlled access and are **not** included in this repository. Registered users must obtain the data from the CHARLS data custodian and comply with its terms. See `DATA_LAYOUT.md` before running the pipeline.

## Environment

Python 3.12 was used for the reported analysis. Create an isolated environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Analysis order

```bash
python build_2013_sample_info.py
python charls_multistate_clean_5wave.py --stage stage --output .
python charls_cohort_build_5wave.py --output .
python fit_fivewave_ctmc.py --domain pain
python fit_fivewave_ctmc.py --domain function
python finalize_full_fivewave.py --domain pain
python finalize_full_fivewave.py --domain function
python generate_fivewave_figures.py
python build_scirep_fivewave_package.py
```

Model fitting is computationally intensive. The `initial_values/` directory contains aggregate numerical starting values used only to initialise the optimisers; it contains no participant records.

The primary models use `rural_nbs`, defined as rural versus urban community classification, as the model-entry residence covariate. The strict complete 11-item sensitivity analysis determines raw BADL/IADL item completeness before applying the structural-skip recoding used by the primary function-state definition.

## Repository hygiene

The `.gitignore` excludes controlled data and participant-level derived files. Before publishing, add final authors, a repository licence chosen by the authors, and a DOI/citation after archiving. Every script and aggregate output should be independently verified by the authors.
