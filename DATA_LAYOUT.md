# Expected controlled-data layout

CHARLS individual-level data are not included. After obtaining authorised data, create `stage/` with one subfolder per wave:

```text
stage/
  2011/
  2013/
  2015/
  2018/
  2020/
```

Place the harmonised Stata modules used by the scripts inside each wave folder. The 2013 folder must include the Health Status and Functioning, Demographic Background, Weights, and Exit Interview modules. Exact variable mappings are documented in the manuscript Supplementary Table S1 and in `charls_multistate_clean_5wave.py`.

Never commit `stage/`, `phase1/`, `cohort/`, `models/`, or any participant-level derived file to a public repository.
