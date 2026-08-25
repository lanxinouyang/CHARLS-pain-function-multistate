# Five-wave analysis audit

## Data integration

- Five waves: 2011, 2013, 2015, 2018, 2020.
- Harmonised unique identifiers: 26,347; person-wave rows: 100,715.
- 2013 health IDs: 18,455; union IDs: 19,689; confirmed deaths: 464; health/death conflicts: 0.
- Duplicate harmonised person-wave rows: 0; post-death health records: 0.

## Analytic sample

- People: 21,235; intervals: 62,135; household clusters: 12,577.
- Interval starts: 14,621 (2011), 14,518 (2013), 16,269 (2015), 16,727 (2018).
- Confirmed death endpoints: 1,973.

## Final model

- Continuous-time four-state panel likelihood with all nine directed living/death transitions.
- Four period-specific baseline-intensity sets; transition-specific exposure and confounder effects.
- 126 parameters; household-cluster robust covariance; 5,000 probability draws.
- Maximum projected gradient: 0.000490 (pain), 0.000725 (function).
- Positive numerical Hessians and positive robust probability covariance; no eigenvalue clipping.

## Focal results

- Pain model, P0→P2, F2 vs F0: HR 1.892978 (95% CI 1.571012-2.280928), P=1.9589e-11.
- Pain model, P0→P2, F1 vs F0: HR 1.074800 (95% CI 0.835488-1.382660), P=0.5746.
- Function model, F0→F2, P1 vs P0: HR 1.547984 (95% CI 1.278623-1.874091), P=7.466e-06.
- Function model, F0→F2, P2 vs P0: HR 2.192971 (95% CI 1.856925-2.589832), P=2.183e-20.

## Interpretation guardrails

These are conditional observational associations. They do not establish causal direction, within-wave temporal order, individual prognosis or intervention effects. The analysis was designed and fitted using all five waves together. Excluding the 2013 observations was used only to assess sensitivity to the observation schedule and sample composition.

## Corrected strict complete 11-item sensitivity analysis

- Raw completeness was assessed before structural-skip BADL recoding.
- People: 17,682; intervals: 42,468.
- Pain model, P0→P2, F2 vs F0: HR 1.550409.
- Function model, F0→F2, P2 vs P0: HR 1.699715.
