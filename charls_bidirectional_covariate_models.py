#!/usr/bin/env python3
"""Fit bidirectional covariate CTMC models for CHARLS 2011--2020.

The opposite domain is a time-varying interval-start exposure.  Demographic
and chronic-disease covariates are frozen at model entry.  The fully connected
three-living-state structure selected by the null-model likelihood-ratio tests
is retained, with death as an absorbing state.

Parameterization
----------------
* Nine transition-specific baseline log intensities.
* Two transition-specific exposure coefficients (level 1 and level 2 versus
  level 0) for each of the nine transitions.
* In the main-adjusted model, confounder coefficients are shared within three
  clinically interpretable transition families: deterioration, recovery, and
  death.  This avoids an unstable 99-parameter model while leaving the main
  bidirectional exposure effects transition-specific.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize


DEFAULT_INPUT = PROJECT / "cohort" / "CHARLS_分析长格式_45岁及以上.csv"
DEFAULT_OUTPUT = PROJECT / "legacy_covariate_outputs"
ACTIVE_N = 3
DEATH_CODE = 3
ALIVE_UNKNOWN_CODE = 4
WAVE_PAIRS = [(2011, 2015), (2015, 2018), (2018, 2020)]
EDUCATION_LEVELS = ["无正规教育", "小学及以下", "初中", "高中及以上"]
EDUCATION_REFERENCE = "高中及以上"
CONFOUNDER_NAMES = [
    "年龄（每10岁）",
    "女性（对男性）",
    "无正规教育（对高中及以上）",
    "小学及以下（对高中及以上）",
    "初中（对高中及以上）",
    "已婚/伴侣（对其他）",
    "农村（对城镇）",
    "慢性病数量（每增加1种）",
]

# The selected null structure: all directed transitions among the three living
# states plus a transition to death from every living state.
EDGES = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (0, 3), (1, 3), (2, 3)]
FAMILY_LABELS = ["恶化", "改善", "死亡"]
EDGE_FAMILY = np.asarray([0, 0, 1, 0, 1, 1, 2, 2, 2], dtype=int)


@dataclass
class DomainSpec:
    name: str
    outcome_col: str
    exposure_col: str
    state_labels: list[str]
    exposure_labels: list[str]
    outcome_cn: str
    exposure_cn: str


@dataclass
class GroupedLikelihood:
    durations: np.ndarray
    exposure: np.ndarray
    confounders: np.ndarray
    counts: np.ndarray
    alive_unknown: np.ndarray

    @property
    def n_groups(self) -> int:
        return len(self.durations)


DOMAIN_SPECS = {
    "pain": DomainSpec(
        name="pain",
        outcome_col="pain_state_with_death",
        exposure_col="function_state",
        state_labels=["P0", "P1", "P2", "D"],
        exposure_labels=["F0", "F1", "F2"],
        outcome_cn="疼痛负担",
        exposure_cn="功能失能",
    ),
    "function": DomainSpec(
        name="function",
        outcome_col="function_state",
        exposure_col="pain_state_with_death",
        state_labels=["F0", "F1", "F2", "D"],
        exposure_labels=["P0", "P1", "P2"],
        outcome_cn="功能失能",
        exposure_cn="疼痛负担",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--hessian", choices=["lbfgs", "numerical"], default="numerical")
    parser.add_argument("--resume-hessian", action="store_true", help="Reuse saved converged parameters and recompute inference only")
    return parser.parse_args()


def valid_alive(row: pd.Series) -> bool:
    if int(row.get("death_confirmed", 0) or 0) == 1:
        return False
    if int(row.get("health_record_present", 0) or 0) == 1:
        return True
    died_raw = pd.to_numeric(pd.Series([row.get("died_raw")]), errors="coerce").iloc[0]
    return int(row.get("sample_record_present", 0) or 0) == 1 and died_raw == 0


def confounder_vector(row: pd.Series) -> np.ndarray | None:
    education = row["education_cat"]
    raw = [row["age_years"], row["female"], row["partnered"], row["rural_nbs"], row["chronic_n"]]
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)) or education not in EDUCATION_LEVELS:
        return None
    age, female, partnered, rural, chronic = numeric
    if not (45 <= age <= 120 and female in (0, 1) and partnered in (0, 1) and rural in (0, 1) and 0 <= chronic <= 14):
        return None
    education_dummies = [float(education == level) for level in EDUCATION_LEVELS[:-1]]
    # Centering/scaling changes numerical conditioning only.  Reported age HR
    # remains per 10 years and chronic-disease HR is transformed back to per
    # one condition below.
    return np.asarray([(age - 60.0) / 10.0, female, *education_dummies, partnered, rural, (chronic - 2.0) / 2.0], dtype=float)


def build_intervals(long: pd.DataFrame, spec: DomainSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    active_outcome = set(spec.state_labels[:ACTIVE_N])
    active_exposure = set(spec.exposure_labels)
    outcome_code = {state: index for index, state in enumerate(spec.state_labels[:ACTIVE_N])}
    exposure_code = {state: index for index, state in enumerate(spec.exposure_labels)}
    records = long.loc[long["eligible_age45"].eq(1)].copy()
    records["person_id"] = records["person_id"].astype("string")
    records = records.sort_values(["person_id", "wave"])

    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "domain": spec.name,
        "eligible_people": int(records["person_id"].nunique()),
        "no_joint_state_entry": 0,
        "no_complete_baseline_confounders": 0,
        "people_with_model_entry": 0,
        "candidate_intervals_after_entry": 0,
        "excluded_start_outcome_missing": 0,
        "excluded_start_exposure_missing": 0,
        "excluded_destination_vital_unknown": 0,
        "included_intervals": 0,
        "fully_observed_destinations": 0,
        "alive_unknown_destinations": 0,
        "death_destinations": 0,
    }

    for person_id, group in records.groupby("person_id", sort=False):
        group = group.sort_values("wave").drop_duplicates("wave", keep="last").set_index("wave", drop=False)
        entry_wave = None
        entry_z = None
        had_joint_state = False
        for _, candidate in group.iterrows():
            if candidate[spec.outcome_col] in active_outcome and candidate[spec.exposure_col] in active_exposure:
                had_joint_state = True
                candidate_z = confounder_vector(candidate)
                if candidate_z is not None:
                    entry_wave = int(candidate["wave"])
                    entry_z = candidate_z
                    break
        if entry_wave is None:
            if had_joint_state:
                audit["no_complete_baseline_confounders"] += 1
            else:
                audit["no_joint_state_entry"] += 1
            continue
        audit["people_with_model_entry"] += 1

        for start_wave, end_wave in WAVE_PAIRS:
            if start_wave < entry_wave or start_wave not in group.index or end_wave not in group.index:
                continue
            audit["candidate_intervals_after_entry"] += 1
            start = group.loc[start_wave]
            end = group.loc[end_wave]
            if start[spec.outcome_col] not in active_outcome:
                audit["excluded_start_outcome_missing"] += 1
                continue
            if start[spec.exposure_col] not in active_exposure:
                audit["excluded_start_exposure_missing"] += 1
                continue

            destination: int | None = None
            destination_label: str | None = None
            if pd.to_numeric(pd.Series([end["death_confirmed"]]), errors="coerce").iloc[0] == 1:
                destination = DEATH_CODE
                destination_label = "D"
                audit["death_destinations"] += 1
            elif end[spec.outcome_col] in active_outcome:
                destination_label = str(end[spec.outcome_col])
                destination = outcome_code[destination_label]
                audit["fully_observed_destinations"] += 1
            elif valid_alive(end):
                destination = ALIVE_UNKNOWN_CODE
                destination_label = "ALIVE_UNKNOWN"
                audit["alive_unknown_destinations"] += 1
            else:
                audit["excluded_destination_vital_unknown"] += 1
                continue

            start_time = pd.to_numeric(pd.Series([start["time_since_entry_years"]]), errors="coerce").iloc[0]
            end_time = pd.to_numeric(pd.Series([end["time_since_entry_years"]]), errors="coerce").iloc[0]
            duration = float(end_time - start_time) if pd.notna(start_time) and pd.notna(end_time) else np.nan
            if not np.isfinite(duration) or duration <= 0:
                start_date = pd.to_datetime(start["interview_date"], errors="coerce")
                end_date = pd.to_datetime(end["interview_date"], errors="coerce")
                duration = float((end_date - start_date).days / 365.25)
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid duration for {person_id}, {start_wave}->{end_wave}")

            z = np.asarray(entry_z, dtype=float)
            row = {
                "person_id": str(person_id),
                "entry_wave": entry_wave,
                "start_wave": start_wave,
                "end_wave": end_wave,
                "interval_years": round(duration, 6),
                "from_state": str(start[spec.outcome_col]),
                "to_state": destination_label,
                "from_code": outcome_code[str(start[spec.outcome_col])],
                "to_code": destination,
                "exposure_state": str(start[spec.exposure_col]),
                "exposure_code": exposure_code[str(start[spec.exposure_col])],
            }
            for name, value in zip(CONFOUNDER_NAMES, z):
                row[name] = float(value)
            rows.append(row)

    intervals = pd.DataFrame(rows)
    if intervals.empty:
        raise ValueError(f"No analysis intervals for {spec.name}")
    intervals = intervals.sort_values(["person_id", "start_wave"]).reset_index(drop=True)
    audit["included_intervals"] = int(len(intervals))
    audit["included_people"] = int(intervals["person_id"].nunique())
    audit["entry_wave_counts"] = {str(int(key)): int(value) for key, value in intervals.drop_duplicates("person_id")["entry_wave"].value_counts().sort_index().items()}
    audit["start_exposure_counts"] = {str(key): int(value) for key, value in intervals["exposure_state"].value_counts().sort_index().items()}
    audit["origin_counts"] = {str(key): int(value) for key, value in intervals["from_state"].value_counts().sort_index().items()}
    audit["destination_counts"] = {str(key): int(value) for key, value in intervals["to_state"].value_counts().sort_index().items()}
    return intervals, audit


def group_intervals(intervals: pd.DataFrame) -> GroupedLikelihood:
    key_columns = ["interval_years", "exposure_code", *CONFOUNDER_NAMES]
    group_index = pd.MultiIndex.from_frame(intervals[key_columns])
    codes, unique_index = pd.factorize(group_index, sort=False)
    unique = unique_index.to_frame(index=False)
    unique.columns = key_columns
    counts = np.zeros((len(unique), ACTIVE_N, 4), dtype=float)
    alive_unknown = np.zeros((len(unique), ACTIVE_N), dtype=float)
    origin = intervals["from_code"].to_numpy(dtype=int)
    destination = intervals["to_code"].to_numpy(dtype=int)
    fully_observed = destination != ALIVE_UNKNOWN_CODE
    np.add.at(counts, (codes[fully_observed], origin[fully_observed], destination[fully_observed]), 1.0)
    np.add.at(alive_unknown, (codes[~fully_observed], origin[~fully_observed]), 1.0)
    return GroupedLikelihood(
        durations=unique["interval_years"].to_numpy(dtype=float),
        exposure=unique["exposure_code"].to_numpy(dtype=int),
        confounders=unique[CONFOUNDER_NAMES].to_numpy(dtype=float),
        counts=counts,
        alive_unknown=alive_unknown,
    )


def unpack_theta(theta: np.ndarray, adjusted: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    alpha = theta[: len(EDGES)]
    beta = theta[len(EDGES): len(EDGES) + len(EDGES) * 2].reshape(len(EDGES), 2)
    gamma = None
    if adjusted:
        gamma = theta[len(EDGES) + len(EDGES) * 2:].reshape(len(FAMILY_LABELS), len(CONFOUNDER_NAMES))
    return alpha, beta, gamma


def q_from_rates(rates: np.ndarray) -> np.ndarray:
    q = np.zeros((4, 4), dtype=float)
    for rate, (origin, destination) in zip(rates, EDGES):
        q[origin, destination] = rate
    for origin in range(ACTIVE_N):
        q[origin, origin] = -q[origin].sum()
    return q


def batch_expm_and_adjoint(q: np.ndarray, durations: np.ndarray, score_probability: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Vectorized 4x4 matrix exponential and optional Frechet adjoint.

    The divided-difference eigendecomposition is algebraically equivalent to
    scipy.linalg.expm_frechet.  It reduces tens of thousands of Python-level
    matrix-exponential calls to batched NumPy linear algebra.
    """
    a = q * durations[:, None, None]
    eigenvalues, eigenvectors = np.linalg.eig(a)
    inverse_vectors = np.linalg.inv(eigenvectors)
    exp_eigenvalues = np.exp(eigenvalues)
    probability_complex = (eigenvectors * exp_eigenvalues[:, None, :]) @ inverse_vectors
    probability = probability_complex.real
    if np.max(np.abs(probability_complex.imag)) > 1e-7:
        raise FloatingPointError("Matrix exponential has non-negligible imaginary component")
    if score_probability is None:
        return probability, None

    delta = eigenvalues[:, :, None] - eigenvalues[:, None, :]
    small = np.abs(delta) < 1e-7
    ratio = np.empty_like(delta)
    # Direct divided differences avoid expm1(delta) overflow when two
    # non-positive eigenvalues are far apart during line-search trial steps.
    exp_i = exp_eigenvalues[:, :, None]
    exp_j = exp_eigenvalues[:, None, :]
    numerator = exp_i - exp_j
    exp_j_broadcast = np.broadcast_to(exp_j, delta.shape)
    ratio[~small] = numerator[~small] / delta[~small]
    ratio[small] = exp_j_broadcast[small] * (
        1 + delta[small] / 2 + delta[small] ** 2 / 6 + delta[small] ** 3 / 24
    )
    divided_difference = ratio

    vectors_t = eigenvectors.transpose(0, 2, 1)
    inverse_t = inverse_vectors.transpose(0, 2, 1)
    middle = vectors_t @ score_probability @ inverse_t
    score_a_complex = inverse_t @ (divided_difference * middle) @ vectors_t
    score_q_complex = durations[:, None, None] * score_a_complex
    if np.max(np.abs(score_q_complex.imag)) > 1e-6:
        raise FloatingPointError("Frechet adjoint has non-negligible imaginary component")
    return probability, score_q_complex.real


def objective_and_gradient(theta: np.ndarray, data: GroupedLikelihood, adjusted: bool) -> tuple[float, np.ndarray]:
    alpha, beta, gamma = unpack_theta(theta, adjusted)
    exposure = np.column_stack((data.exposure == 1, data.exposure == 2)).astype(float)
    eta = alpha[None, :] + exposure @ beta.T
    if adjusted and gamma is not None:
        family_effect = data.confounders @ gamma.T
        eta += family_effect[:, EDGE_FAMILY]
    rates = np.exp(eta)
    q = np.zeros((data.n_groups, 4, 4), dtype=float)
    for edge_index, (origin, destination) in enumerate(EDGES):
        q[:, origin, destination] = rates[:, edge_index]
    q[:, np.arange(ACTIVE_N), np.arange(ACTIVE_N)] = -q[:, :ACTIVE_N, :].sum(axis=2)
    probability, _ = batch_expm_and_adjoint(q, data.durations)
    observed = data.counts > 0
    selected_probability = probability[:, :ACTIVE_N, :][observed]
    if np.any(~np.isfinite(selected_probability)) or np.any(selected_probability <= 1e-300):
        return 1e100, np.zeros_like(theta)
    log_likelihood = float(np.sum(data.counts[observed] * np.log(selected_probability)))
    score_probability = np.zeros_like(probability)
    score_probability[:, :ACTIVE_N, :][observed] = data.counts[observed] / selected_probability

    survival_probability = probability[:, :ACTIVE_N, :ACTIVE_N].sum(axis=2)
    alive_mask = data.alive_unknown > 0
    if np.any(~np.isfinite(survival_probability[alive_mask])) or np.any(survival_probability[alive_mask] <= 1e-300):
        return 1e100, np.zeros_like(theta)
    log_likelihood += float(np.sum(data.alive_unknown[alive_mask] * np.log(survival_probability[alive_mask])))
    alive_score = np.divide(
        data.alive_unknown,
        survival_probability,
        out=np.zeros_like(data.alive_unknown),
        where=alive_mask,
    )
    score_probability[:, :ACTIVE_N, :ACTIVE_N] += alive_score[:, :, None]
    _, score_q = batch_expm_and_adjoint(q, data.durations, score_probability)
    assert score_q is not None
    edge_score = np.column_stack([
        rates[:, edge_index] * (score_q[:, origin, destination] - score_q[:, origin, origin])
        for edge_index, (origin, destination) in enumerate(EDGES)
    ])
    gradient_alpha = edge_score.sum(axis=0)
    gradient_beta = edge_score.T @ exposure
    gradient_parts = [gradient_alpha.ravel(), gradient_beta.ravel()]
    if adjusted and gamma is not None:
        gradient_gamma = np.zeros_like(gamma)
        for family in range(len(FAMILY_LABELS)):
            gradient_gamma[family] = edge_score[:, EDGE_FAMILY == family].sum(axis=1) @ data.confounders
        gradient_parts.append(gradient_gamma.ravel())
    return -log_likelihood, -np.concatenate(gradient_parts)


def run_self_test() -> None:
    rng = np.random.default_rng(20260820)
    rates = np.exp(rng.normal(-2, 0.4, size=(5, len(EDGES))))
    q = np.zeros((5, 4, 4))
    for edge_index, (origin, destination) in enumerate(EDGES):
        q[:, origin, destination] = rates[:, edge_index]
    q[:, np.arange(ACTIVE_N), np.arange(ACTIVE_N)] = -q[:, :ACTIVE_N, :].sum(axis=2)
    durations = rng.uniform(1.5, 4.5, size=5)
    score_probability = rng.uniform(0.1, 2, size=(5, 4, 4))
    probability, score_q = batch_expm_and_adjoint(q, durations, score_probability)
    maximum_probability_error = 0.0
    maximum_adjoint_error = 0.0
    for index in range(5):
        a = q[index] * durations[index]
        expected_probability = expm(a)
        expected_score_q = durations[index] * expm_frechet(a.T, score_probability[index], compute_expm=False)
        maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(probability[index] - expected_probability))))
        maximum_adjoint_error = max(maximum_adjoint_error, float(np.max(np.abs(score_q[index] - expected_score_q))))
    if maximum_probability_error > 1e-9 or maximum_adjoint_error > 1e-8:
        raise AssertionError((maximum_probability_error, maximum_adjoint_error))
    print(json.dumps({"batch_expm_max_error": maximum_probability_error, "adjoint_max_error": maximum_adjoint_error}, indent=2))


def null_alpha(spec: DomainSpec) -> np.ndarray:
    rate_path = Path(__file__).resolve().parent / spec.name / "转移强度.csv"
    rates = pd.read_csv(rate_path)
    lookup = {(row.from_state, row.to_state): float(row.intensity_per_year) for row in rates.itertuples()}
    return np.log(np.asarray([lookup[(spec.state_labels[o], spec.state_labels[d])] for o, d in EDGES]))


def numerical_hessian_from_gradient(function, optimum: np.ndarray, step: float = 2e-4) -> np.ndarray:
    n = len(optimum)
    hessian = np.zeros((n, n), dtype=float)
    for index in range(n):
        plus = optimum.copy(); plus[index] += step
        minus = optimum.copy(); minus[index] -= step
        gradient_plus = function(plus)[1]
        gradient_minus = function(minus)[1]
        hessian[:, index] = (gradient_plus - gradient_minus) / (2 * step)
    return (hessian + hessian.T) / 2


def fit_model(
    spec: DomainSpec,
    data: GroupedLikelihood,
    adjusted: bool,
    maxiter: int,
    hessian_method: str,
    start_from: np.ndarray | None = None,
    fixed_theta: np.ndarray | None = None,
) -> dict[str, Any]:
    model_label = "主调整" if adjusted else "最小调整"
    n_parameters = len(EDGES) + len(EDGES) * 2 + (len(FAMILY_LABELS) * len(CONFOUNDER_NAMES) if adjusted else 0)
    if start_from is None:
        initial = np.concatenate([null_alpha(spec), np.zeros(len(EDGES) * 2), np.zeros(n_parameters - len(EDGES) * 3)])
    else:
        initial = np.zeros(n_parameters)
        initial[: len(start_from)] = start_from
    objective = lambda value: objective_and_gradient(value, data, adjusted)
    bounds = [(math.log(1e-6), math.log(5.0))] * len(EDGES)
    bounds += [(-3.0, 3.0)] * (len(EDGES) * 2)
    if adjusted:
        for _ in FAMILY_LABELS:
            bounds.extend([(-1.5, 1.5), *[(-2.0, 2.0)] * 6, (-1.0, 1.0)])
    started = time.time()
    fit = None
    if fixed_theta is None:
        fit = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-6, "maxls": 100, "maxcor": 40},
        )
        optimum = np.asarray(fit.x, dtype=float)
    else:
        optimum = np.asarray(fixed_theta, dtype=float)
        if len(optimum) != n_parameters:
            raise ValueError(f"Saved parameter length {len(optimum)} != expected {n_parameters}")
    value, gradient = objective(optimum)
    if hessian_method == "numerical":
        hessian = numerical_hessian_from_gradient(objective, optimum)
        covariance = np.linalg.pinv(hessian, rcond=1e-10)
        hessian_source = "central difference of analytic score"
    else:
        if fit is None:
            raise ValueError("L-BFGS Hessian requires a newly optimized fit")
        covariance = np.asarray(fit.hess_inv.todense())
        hessian = np.linalg.pinv(covariance, rcond=1e-10)
        hessian_source = "L-BFGS inverse-Hessian approximation"
    covariance = (covariance + covariance.T) / 2
    se = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    alpha, beta, gamma = unpack_theta(optimum, adjusted)

    exposure_rows: list[dict[str, Any]] = []
    beta_offset = len(EDGES)
    for edge_index, (origin, destination) in enumerate(EDGES):
        for exposure_index, exposure_level in enumerate(spec.exposure_labels[1:]):
            parameter_index = beta_offset + edge_index * 2 + exposure_index
            estimate = beta[edge_index, exposure_index]
            standard_error = se[parameter_index]
            exposure_rows.append({
                "domain": spec.name,
                "model": model_label,
                "from_state": spec.state_labels[origin],
                "to_state": spec.state_labels[destination],
                "transition_family": FAMILY_LABELS[EDGE_FAMILY[edge_index]],
                "exposure": spec.exposure_cn,
                "contrast": f"{exposure_level} vs {spec.exposure_labels[0]}",
                "log_hr": float(estimate),
                "se_log_hr": float(standard_error),
                "hr": float(math.exp(estimate)),
                "ci95_low": float(math.exp(estimate - 1.96 * standard_error)),
                "ci95_high": float(math.exp(estimate + 1.96 * standard_error)),
                "p_value_wald": float(math.erfc(abs(estimate / standard_error) / math.sqrt(2))) if standard_error > 0 else np.nan,
            })

    confounder_rows: list[dict[str, Any]] = []
    if adjusted and gamma is not None:
        gamma_offset = len(EDGES) * 3
        for family_index, family in enumerate(FAMILY_LABELS):
            for covariate_index, covariate in enumerate(CONFOUNDER_NAMES):
                parameter_index = gamma_offset + family_index * len(CONFOUNDER_NAMES) + covariate_index
                scale = 2.0 if covariate_index == len(CONFOUNDER_NAMES) - 1 else 1.0
                estimate = gamma[family_index, covariate_index] / scale
                standard_error = se[parameter_index] / scale
                confounder_rows.append({
                    "domain": spec.name,
                    "model": model_label,
                    "transition_family": family,
                    "covariate": covariate,
                    "log_hr": float(estimate),
                    "se_log_hr": float(standard_error),
                    "hr": float(math.exp(estimate)),
                    "ci95_low": float(math.exp(estimate - 1.96 * standard_error)),
                    "ci95_high": float(math.exp(estimate + 1.96 * standard_error)),
                    "p_value_wald": float(math.erfc(abs(estimate / standard_error) / math.sqrt(2))) if standard_error > 0 else np.nan,
                })

    summary = {
        "domain": spec.name,
        "model": model_label,
        "converged": bool(fit.success) if fit is not None else bool(np.max(np.abs(gradient)) < 0.005),
        "message": str(fit.message) if fit is not None else "Reused converged L-BFGS parameters; inference recomputed",
        "n_parameters": n_parameters,
        "negative_log_likelihood": float(value),
        "log_likelihood": float(-value),
        "aic": float(2 * value + 2 * n_parameters),
        "iterations": int(fit.nit) if fit is not None else 0,
        "function_evaluations": int(fit.nfev) if fit is not None else 1,
        "max_abs_gradient": float(np.max(np.abs(gradient))),
        "hessian_source": hessian_source,
        "smallest_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
        "hessian_condition_number": float(np.linalg.cond(hessian)),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    return {
        "summary": summary,
        "theta": optimum,
        "covariance": covariance,
        "exposure_rows": exposure_rows,
        "confounder_rows": confounder_rows,
    }


def write_method_report(
    output: Path,
    audits: dict[str, Any],
    summaries: dict[str, Any],
    exposure_rows: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# CHARLS 双向协变量连续时间多状态模型报告",
        "",
        "## 模型设定",
        "",
        "疼痛模型以区间起点功能状态为时变暴露；功能模型以区间起点疼痛状态为时变暴露。两者均采用三个活态完全双向转移、D为吸收态的连续时间Markov结构。年龄、性别、教育、婚姻、城乡及慢病数固定于模型入组波次。",
        "",
        "对向状态的效应对九条转移分别估计。主调整模型中的混杂因素效应在恶化、改善及死亡三类转移内共享，以保持可识别性；因此对向状态HR是转移特异的，混杂因素HR是转移族特异的。",
        "",
        "## 分析样本",
        "",
    ]
    for domain, audit in audits.items():
        spec = DOMAIN_SPECS[domain]
        lines.extend([
            f"- {spec.outcome_cn}模型：{audit['included_people']:,}人，{audit['included_intervals']:,}个区间；死亡结局{audit['death_destinations']:,}个；已知存活但结局状态缺失{audit['alive_unknown_destinations']:,}个。",
        ])
    lines.extend([
        f"- 23,578名年龄合格者中，203人从无疼痛与功能均可判定的联合入组波次，320人在联合状态可判定波次缺少至少一项主调整变量；另有已入组但无后续可用区间者。因此主模型采用20,111人的完整病例区间。",
    ])
    if summaries:
        lines.extend(["", "## 收敛诊断", ""])
        for domain, domain_summaries in summaries.items():
            for model_name, summary in domain_summaries.items():
                lines.append(
                    f"- {DOMAIN_SPECS[domain].outcome_cn}—{model_name}：收敛={summary['converged']}；参数={summary['n_parameters']}；"
                    f"log-likelihood={summary['log_likelihood']:.2f}；AIC={summary['aic']:.2f}；最大绝对梯度={summary['max_abs_gradient']:.3g}；"
                    f"Hessian最小特征值={summary['smallest_hessian_eigenvalue']:.3g}。"
                )
    if exposure_rows:
        estimates = pd.DataFrame(exposure_rows)
        estimates = estimates.loc[estimates["model"].eq("主调整")]

        def result(domain: str, origin: str, destination: str, contrast: str) -> str:
            selected = estimates.loc[
                estimates["domain"].eq(domain)
                & estimates["from_state"].eq(origin)
                & estimates["to_state"].eq(destination)
                & estimates["contrast"].eq(contrast)
            ].iloc[0]
            return f"HR={selected.hr:.2f}（95%CI {selected.ci95_low:.2f}–{selected.ci95_high:.2f}）"

        lines.extend([
            "",
            "## 主要双向关联",
            "",
            f"- 与F0相比，F1和F2分别与P0→P2强度升高相关：{result('pain', 'P0', 'P2', 'F1 vs F0')}和{result('pain', 'P0', 'P2', 'F2 vs F0')}。",
            f"- 与F0相比，F2与P2→P0和P2→P1改善强度降低相关：{result('pain', 'P2', 'P0', 'F2 vs F0')}和{result('pain', 'P2', 'P1', 'F2 vs F0')}。",
            f"- 与P0相比，P1和P2分别与F0→F1强度升高相关：{result('function', 'F0', 'F1', 'P1 vs P0')}和{result('function', 'F0', 'F1', 'P2 vs P0')}。",
            f"- 与P0相比，P1和P2分别与F0→F2强度升高相关：{result('function', 'F0', 'F2', 'P1 vs P0')}和{result('function', 'F0', 'F2', 'P2 vs P0')}。",
            "- 疼痛负担与已经进入F1/F2后的改善转移未见明确关联；该结果不等于证明不存在影响，需要结合联合状态模型和敏感性分析判断。",
        ])
    lines.extend([
        "",
        "## 解释边界",
        "",
        "HR表示给定模型假设和调整变量后的转移强度比，不等同于因果效应。对向状态在每个区间起点测量并在该区间内视为固定，因此这是滞后区间暴露模型，不等同于同时估计两个过程的联合十状态模型。由于访视间隔较长，P0↔P2或F0↔F2的直接强度代表两次观测之间模型所需的净跃迁通道，不应解释为已观察到确切的瞬时跳变路径。",
        "",
        "当前结果为未加抽样权重的完整病例模型。抑郁、吸烟、饮酒、体力活动、抽样权重及联合十状态模型将在敏感性/扩展分析中处理。",
        "",
    ])
    (output / "CHARLS_双向协变量多状态模型报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(args.input, dtype={"person_id": "string"}, low_memory=False)
    audits: dict[str, Any] = {}
    grouped: dict[str, GroupedLikelihood] = {}
    for domain, spec in DOMAIN_SPECS.items():
        intervals, audit = build_intervals(long, spec)
        domain_dir = args.output / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        intervals.to_csv(domain_dir / "完整病例模型区间.csv", index=False)
        grouped[domain] = group_intervals(intervals)
        audit["likelihood_groups"] = grouped[domain].n_groups
        audits[domain] = audit
    (args.output / "双向协变量模型数据质控.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit_only:
        write_method_report(args.output, audits, {})
        print(json.dumps(audits, ensure_ascii=False, indent=2))
        return

    all_summaries: dict[str, Any] = {}
    all_exposure_rows: list[dict[str, Any]] = []
    all_confounder_rows: list[dict[str, Any]] = []
    for domain, spec in DOMAIN_SPECS.items():
        if args.resume_hessian:
            saved = np.load(args.output / domain / "模型参数与协方差.npz")
            minimum = fit_model(
                spec, grouped[domain], False, args.maxiter, "numerical",
                fixed_theta=saved["minimum_theta"],
            )
            main_adjusted = fit_model(
                spec, grouped[domain], True, args.maxiter, "numerical",
                fixed_theta=saved["adjusted_theta"],
            )
        else:
            minimum = fit_model(spec, grouped[domain], False, args.maxiter, args.hessian)
            main_adjusted = fit_model(spec, grouped[domain], True, args.maxiter, args.hessian, minimum["theta"])
        all_summaries[domain] = {
            "最小调整": minimum["summary"],
            "主调整": main_adjusted["summary"],
        }
        all_exposure_rows.extend(minimum["exposure_rows"])
        all_exposure_rows.extend(main_adjusted["exposure_rows"])
        all_confounder_rows.extend(main_adjusted["confounder_rows"])
        domain_dir = args.output / domain
        np.savez_compressed(
            domain_dir / "模型参数与协方差.npz",
            minimum_theta=minimum["theta"],
            minimum_covariance=minimum["covariance"],
            adjusted_theta=main_adjusted["theta"],
            adjusted_covariance=main_adjusted["covariance"],
        )
    exposure_frame = pd.DataFrame(all_exposure_rows)
    exposure_frame.to_csv(args.output / "双向转移特异HR.csv", index=False)
    exposure_frame.loc[
        exposure_frame["model"].eq("主调整") & exposure_frame["to_state"].ne("D")
    ].to_csv(args.output / "主调整模型双向活态转移HR.csv", index=False)
    pd.DataFrame(all_confounder_rows).to_csv(args.output / "主调整模型混杂因素HR.csv", index=False)
    (args.output / "全部协变量模型摘要.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_method_report(args.output, audits, all_summaries, all_exposure_rows)
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
