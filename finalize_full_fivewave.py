#!/usr/bin/env python3
"""Finalize inference and sensitivities for the 126-parameter five-wave model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import fit_fivewave_ctmc as fit
from scipy.stats import chi2


def projected_gradient(theta: np.ndarray, gradient: np.ndarray, model: str, n_periods: int) -> tuple[float, list[int]]:
    common = fit.E + (n_periods - 1) * fit.E + 2 * fit.E
    bounds = [(math.log(1e-6), math.log(5.0))] * fit.E + [(-4.0, 4.0)] * (common - fit.E)
    bounds += [(-5.0, 5.0)] * (len(theta) - common)
    projected = gradient.copy()
    active = []
    for index, (value, score, (lower, upper)) in enumerate(zip(theta, gradient, bounds)):
        if abs(value - lower) <= 1e-6 and score > 0:
            projected[index] = 0.0; active.append(index)
        elif abs(value - upper) <= 1e-6 and score < 0:
            projected[index] = 0.0; active.append(index)
    return float(np.max(np.abs(projected))), active


def fit_sensitivity(domain: str, analysis: str, intervals: pd.DataFrame, data: fit.PeriodData,
                    initial: np.ndarray, n_periods: int, maxiter: int, extra: dict | None = None) -> tuple[dict, np.ndarray]:
    theta, audit = fit.fit_model(data, "full", n_periods, initial, maxiter)
    nll, gradient = fit.objective(theta, "full", n_periods, data)
    max_projected, active = projected_gradient(theta, gradient, "full", n_periods)
    row = {
        "analysis": analysis, **fit.extract_severe(domain, theta, n_periods, "full"),
        "people": int(intervals.person_id.nunique()), "intervals": int(len(intervals)),
        **audit, "max_abs_projected_gradient": max_projected, "active_bound_indices": ",".join(map(str, active)),
    }
    if extra: row.update(extra)
    return row, theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["pain", "function"], required=True)
    parser.add_argument("--maxiter", type=int, default=900)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    domain = args.domain
    out = fit.FIVE / "models" / domain
    intervals = pd.read_csv(out / "fivewave_primary_intervals.csv", dtype={"person_id": "string", "household_id": "string", "cluster_id": "string"})
    data = fit.group_intervals(intervals)
    nested = np.load(out / "fivewave_nested_parameters.npz")
    full_start = np.asarray(nested["full"])
    full_theta, refinement = fit.fit_model(data, "full", 4, full_start, args.maxiter)
    nll, gradient = fit.objective(full_theta, "full", 4, data)
    max_projected, active = projected_gradient(full_theta, gradient, "full", 4)

    hessian = fit.numerical_hessian(full_theta, data, 4, "full")
    model_covariance = np.linalg.pinv(hessian, rcond=1e-10)
    model_covariance = (model_covariance + model_covariance.T) / 2
    robust, robust_audit = fit.robust_covariance(full_theta, model_covariance, intervals, 4, "full")
    hrs = fit.hr_rows(domain, full_theta, robust, 4, "full")
    probabilities, probability_audit = fit.probability_rows(domain, full_theta, robust, intervals, 4, args.draws, "full")
    np.savez_compressed(out / "fivewave_final_full_parameters.npz", theta=full_theta, hessian=hessian,
                        model_covariance=model_covariance, household_robust_covariance=robust)
    pd.DataFrame(hrs).to_csv(out / "fivewave_final_full_household_robust_hr.csv", index=False)
    pd.DataFrame(probabilities).to_csv(out / "fivewave_final_full_period_probabilities.csv", index=False)

    original_audit = json.loads((out / "fivewave_primary_audit.json").read_text(encoding="utf-8"))
    shared_nll = original_audit["shared_fit"]["negative_log_likelihood"]
    primary_nll = original_audit["primary_fit"]["negative_log_likelihood"]
    comparisons = {
        "shared_78_vs_full_126": {
            "chi_square": float(2 * (shared_nll - nll)), "df": 48,
            "p_value": float(chi2.sf(2 * (shared_nll - nll), 48)),
        },
        "primary_110_vs_full_126": {
            "chi_square": float(2 * (primary_nll - nll)), "df": 16,
            "p_value": float(chi2.sf(2 * (primary_nll - nll), 16)),
        },
    }
    final_audit = {
        "domain": domain, "final_model": "126 parameters; all nine transitions have distinct confounder coefficients",
        "people": int(intervals.person_id.nunique()), "intervals": int(len(intervals)),
        "household_clusters": int(intervals.cluster_id.nunique()), "negative_log_likelihood": float(nll),
        "refinement": refinement, "max_abs_gradient": float(np.max(np.abs(gradient))),
        "max_abs_projected_gradient": max_projected, "active_bound_indices": active,
        "comparisons": comparisons, "hessian_minimum_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
        "hessian_condition_number": float(np.linalg.cond(hessian)), "robust_covariance": robust_audit,
        "probability_covariance": probability_audit, "probability_draws": args.draws,
    }
    (out / "fivewave_final_full_audit.json").write_text(json.dumps(final_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    long = pd.read_csv(fit.FIVE / "cohort" / "CHARLS_分析长格式_45岁及以上.csv", dtype={"person_id": "string", "household_id": "string"}, low_memory=False)
    sensitivity_rows = []
    weight_values, weight_audit = fit.sampling_weights(intervals)
    weighted_data = fit.group_intervals(intervals, weight_values)
    row, _ = fit_sensitivity(domain, "cross-sectional-weighted", intervals.loc[weight_values.notna()].copy(),
                             weighted_data, full_theta, 4, args.maxiter,
                             {"missing_weights": int(weight_values.isna().sum())})
    sensitivity_rows.append(row)
    weight_audit["fit"] = row
    (out / "fivewave_final_weighted_audit.json").write_text(json.dumps(weight_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    strict_intervals, strict_build = fit.prepare_intervals(domain, long, strict_function=True)
    strict_data = fit.group_intervals(strict_intervals)
    row, _ = fit_sensitivity(domain, "strict-complete-11-item-function", strict_intervals, strict_data,
                             full_theta, 4, args.maxiter)
    sensitivity_rows.append(row)
    (out / "fivewave_final_strict_function_audit.json").write_text(json.dumps({"interval_build": strict_build, "fit": row}, ensure_ascii=False, indent=2), encoding="utf-8")

    exit_ids = pd.read_stata(fit.FIVE / "stage" / "2013" / "Exit_Interview.dta", columns=["ID"], convert_categoricals=False)["ID"].astype("string").str.replace(r"\.0$", "", regex=True)
    exit_only = long.copy()
    wave2013 = exit_only.wave.eq(2013)
    weight_only_death = wave2013 & exit_only.death_confirmed.eq(1) & ~exit_only.person_id.astype("string").isin(set(exit_ids.dropna()))
    exit_only.loc[weight_only_death, "death_confirmed"] = 0
    exit_only.loc[weight_only_death, "died_raw"] = 0
    for column in ["pain_state_with_death", "function_state", "function_state_complete11", "joint_state"]:
        exit_only.loc[weight_only_death, column] = pd.NA
    death_intervals, death_build = fit.prepare_intervals(domain, exit_only)
    death_data = fit.group_intervals(death_intervals)
    row, _ = fit_sensitivity(domain, "2013-exit-interview-deaths-only", death_intervals, death_data,
                             full_theta, 4, args.maxiter, {"weight_only_deaths_reclassified": int(weight_only_death.sum())})
    sensitivity_rows.append(row)
    (out / "fivewave_final_exit_death_audit.json").write_text(json.dumps({"interval_build": death_build, "fit": row}, ensure_ascii=False, indent=2), encoding="utf-8")

    no2013 = long.loc[long.wave.ne(2013)].copy()
    leave_pairs = ((2011, 2015), (2015, 2018), (2018, 2020))
    leave_intervals, leave_build = fit.prepare_intervals(domain, no2013, (2011, 2015, 2018), leave_pairs)
    leave_data = fit.group_intervals(leave_intervals)
    leave_initial = fit.initial_from_old(domain, "full", 3)
    row, _ = fit_sensitivity(domain, "leave-2013-out", leave_intervals, leave_data,
                             leave_initial, 3, args.maxiter)
    sensitivity_rows.append(row)
    (out / "fivewave_final_leave_2013_out_audit.json").write_text(json.dumps({"interval_build": leave_build, "fit": row}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(sensitivity_rows).to_csv(out / "fivewave_final_sensitivity_summary.csv", index=False)
    print(json.dumps({"final": final_audit, "sensitivities": sensitivity_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
