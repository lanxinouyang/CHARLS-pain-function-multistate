# Five-wave CHARLS pain-function multistate analysis

Reproducible analysis code for the study **Bidirectional transitions between musculoskeletal pain burden and functional limitations in middle-aged and older Chinese adults: a five-wave multistate cohort study**.

## What is included

- harmonisation and quality control for 2011, 2013, 2015, 2018, and 2020;
- wave-specific construction and audit of the 2013 sample-information file required by the public-release layout;
- analytic cohort and adjacent-wave interval construction;
- continuous-time multistate models for pain and function, household-cluster robust inference, model-structure tests, and sensitivity analyses;
- publication-figure generation code;
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
python charls_multistate_clean_5wave.py --stage stage --output phase1
python charls_cohort_build_5wave.py \
  --stage stage \
  --phase1 phase1/CHARLS_统一长格式数据_2011_2020_5wave.csv \
  --output cohort
python fit_fivewave_ctmc.py --domain pain --skip-sensitivities
python fit_fivewave_ctmc.py --domain function --skip-sensitivities
python finalize_full_fivewave.py --domain pain
python finalize_full_fivewave.py --domain function
python generate_fivewave_figures.py
```

The two `fit_fivewave_ctmc.py` commands estimate the nested 78-, 110-, and
126-parameter structures. The two `finalize_full_fivewave.py` commands refine
the selected 126-parameter models, calculate household-robust inference and
probabilities, and run the four sensitivity analyses. `--skip-sensitivities`
avoids first running a redundant set under the intermediate 110-parameter
model.

Model fitting is computationally intensive. The `initial_values/` directory contains aggregate numerical starting values used only to initialise the optimisers; it contains no participant records.

`generate_fivewave_figures.py` can run in either of two modes. With authorised
data and model outputs present, it refreshes `source_data/` and the figures.
Without controlled data, it rebuilds the figures directly from the supplied
non-disclosive files in `source_data/`.

Run the lightweight regression tests with:

```bash
python -m unittest discover -s tests
```

Journal-submission documents are maintained separately from the reproducible
analysis pipeline. This repository does not generate a journal submission
package; the manuscript submitted to BMC Geriatrics must use the title and
files supplied in the BMC submission package.

The primary models use `rural_nbs`, defined as rural versus urban community classification, as the model-entry residence covariate. The strict complete 11-item sensitivity analysis determines raw BADL/IADL item completeness before applying the structural-skip recoding used by the primary function-state definition.

## Repository hygiene

The `.gitignore` excludes controlled data and participant-level derived files. Licence and citation information for reuse should be taken from the corresponding public repository and archived release. Every script and aggregate output should be independently verified by users.
