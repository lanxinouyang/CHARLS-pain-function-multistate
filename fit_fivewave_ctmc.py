#!/usr/bin/env python3
"""Fit audited five-wave CHARLS continuous-time multistate models.

Primary structure: transition-specific exposure coefficients; confounder
coefficients specific to each of six living transitions and shared across the
three death transitions. Period-specific baseline intensities are estimated for
2011-2013, 2013-2015, 2015-2018, and 2018-2020.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIVE = ROOT
sys.path.insert(0, str(ROOT))

from scipy.optimize import minimize
from scipy.stats import chi2
import charls_bidirectional_covariate_models as legacy


E = len(legacy.EDGES)
K = len(legacy.CONFOUNDER_NAMES)
PRIMARY_PERIODS = (2011, 2013, 2015, 2018)
PRIMARY_PAIRS = ((2011, 2013), (2013, 2015), (2015, 2018), (2018, 2020))
PERIOD_LABELS = {
    2011: "2011-2013", 2013: "2013-2015", 2015: "2015-2018", 2018: "2018-2020"
}


@dataclass
class PeriodData:
    durations: np.ndarray
    periods: np.ndarray
    exposure: np.ndarray
    confounders: np.ndarray
    counts: np.ndarray
    alive_unknown: np.ndarray

    @property
    def n_groups(self) -> int:
        return len(self.durations)


def prepare_intervals(
    domain: str,
    long: pd.DataFrame,
    period_starts: tuple[int, ...] = PRIMARY_PERIODS,
    wave_pairs: tuple[tuple[int, int], ...] = PRIMARY_PAIRS,
    strict_function: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if domain == "pain":
        spec = legacy.DomainSpec(
            name="pain", outcome_col="pain_state_with_death",
            exposure_col="function_state_complete11" if strict_function else "function_state",
            state_labels=["P0", "P1", "P2", "D"], exposure_labels=["F0", "F1", "F2"],
            outcome_cn="pain", exposure_cn="function",
        )
    else:
        spec = legacy.DomainSpec(
            name="function", outcome_col="function_state_complete11" if strict_function else "function_state",
            exposure_col="pain_state_with_death",
            state_labels=["F0", "F1", "F2", "D"], exposure_labels=["P0", "P1", "P2"],
            outcome_cn="function", exposure_cn="pain",
        )
    legacy.WAVE_PAIRS = list(wave_pairs)
    intervals, audit = legacy.build_intervals(long, spec)
    starts = long.rename(columns={"wave": "start_wave", "age_years": "age_at_interval_start"})
    starts = starts[["person_id", "start_wave", "age_at_interval_start", "household_id", "msk_pain_n", "individual_weight"]]
    intervals = intervals.merge(starts, on=["person_id", "start_wave"], how="left", validate="many_to_one")
    if intervals["age_at_interval_start"].isna().any():
        raise ValueError("Missing interval-start age")
    intervals["年龄（每10岁）"] = (pd.to_numeric(intervals["age_at_interval_start"]) - 60.0) / 10.0
    intervals["cluster_id"] = intervals["household_id"].fillna("P:" + intervals["person_id"].astype(str))
    code = {wave: index for index, wave in enumerate(period_starts)}
    intervals["period_code"] = intervals["start_wave"].map(code)
    if intervals["period_code"].isna().any():
        raise ValueError("Unexpected interval start wave")
    intervals["period_code"] = intervals["period_code"].astype(int)
    return intervals, audit


def sampling_weights(intervals: pd.DataFrame) -> tuple[pd.Series, dict]:
    raw = pd.to_numeric(intervals["individual_weight"], errors="coerce")
    result = pd.Series(np.nan, index=intervals.index, dtype=float)
    audit = {"missing_weights": int(raw.isna().sum()), "periods": {}}
    for start_wave, index in intervals.groupby("start_wave").groups.items():
        values = raw.loc[index]
        valid = values.dropna()
        if valid.empty:
            continue
        low, high = valid.quantile([0.01, 0.99])
        trimmed = values.clip(lower=low, upper=high)
        normalized = trimmed / trimmed.mean()
        result.loc[index] = normalized
        audit["periods"][str(start_wave)] = {
            "available": int(valid.size), "p01": float(low), "p99": float(high),
            "normalization_mean": float(trimmed.mean()),
        }
    return result, audit


def group_intervals(intervals: pd.DataFrame, case_weights: pd.Series | None = None) -> PeriodData:
    keys = ["period_code", "interval_years", "exposure_code", *legacy.CONFOUNDER_NAMES]
    working = intervals.copy()
    weights = np.ones(len(working), dtype=float) if case_weights is None else case_weights.to_numpy(dtype=float)
    keep = np.isfinite(weights) & (weights > 0)
    working = working.loc[keep].reset_index(drop=True)
    weights = weights[keep]
    group_index = pd.MultiIndex.from_frame(working[keys])
    codes, unique_index = pd.factorize(group_index, sort=False)
    unique = unique_index.to_frame(index=False)
    unique.columns = keys
    counts = np.zeros((len(unique), 3, 4), dtype=float)
    alive_unknown = np.zeros((len(unique), 3), dtype=float)
    origin = working["from_code"].to_numpy(dtype=int)
    destination = working["to_code"].to_numpy(dtype=int)
    known = destination != legacy.ALIVE_UNKNOWN_CODE
    np.add.at(counts, (codes[known], origin[known], destination[known]), weights[known])
    np.add.at(alive_unknown, (codes[~known], origin[~known]), weights[~known])
    return PeriodData(
        durations=unique["interval_years"].to_numpy(dtype=float),
        periods=unique["period_code"].to_numpy(dtype=int),
        exposure=unique["exposure_code"].to_numpy(dtype=int),
        confounders=unique[legacy.CONFOUNDER_NAMES].to_numpy(dtype=float),
        counts=counts, alive_unknown=alive_unknown,
    )


def parameter_count(model: str, n_periods: int) -> int:
    common = E + (n_periods - 1) * E + 2 * E
    gamma = {"shared": 3 * K, "primary": 7 * K, "full": E * K}[model]
    return common + gamma


def unpack(theta: np.ndarray, model: str, n_periods: int):
    cursor = 0
    alpha = theta[cursor:cursor + E]; cursor += E
    delta = theta[cursor:cursor + (n_periods - 1) * E].reshape(n_periods - 1, E); cursor += (n_periods - 1) * E
    beta = theta[cursor:cursor + 2 * E].reshape(E, 2); cursor += 2 * E
    if model == "shared":
        gamma = theta[cursor:].reshape(3, K)
    elif model == "primary":
        gamma = (theta[cursor:cursor + 6 * K].reshape(6, K), theta[cursor + 6 * K:].reshape(K))
    else:
        gamma = theta[cursor:].reshape(E, K)
    return alpha, delta, beta, gamma


def linear_predictor(theta: np.ndarray, model: str, n_periods: int, data: PeriodData):
    alpha, delta, beta, gamma = unpack(theta, model, n_periods)
    exposure = np.column_stack((data.exposure == 1, data.exposure == 2)).astype(float)
    eta = alpha[None, :] + exposure @ beta.T
    nonreference = data.periods > 0
    eta[nonreference] += delta[data.periods[nonreference] - 1]
    if model == "shared":
        eta += (data.confounders @ gamma.T)[:, legacy.EDGE_FAMILY]
    elif model == "primary":
        living, death = gamma
        eta[:, :6] += data.confounders @ living.T
        eta[:, 6:] += (data.confounders @ death)[:, None]
    else:
        eta += data.confounders @ gamma.T
    return eta, exposure


def objective(theta: np.ndarray, model: str, n_periods: int, data: PeriodData):
    eta, exposure = linear_predictor(theta, model, n_periods, data)
    rates = np.exp(np.clip(eta, -30, 8))
    q = np.zeros((data.n_groups, 4, 4), dtype=float)
    for edge, (origin, destination) in enumerate(legacy.EDGES):
        q[:, origin, destination] = rates[:, edge]
    q[:, np.arange(3), np.arange(3)] = -q[:, :3, :].sum(axis=2)
    probability, _ = legacy.batch_expm_and_adjoint(q, data.durations)
    observed = data.counts > 0
    selected = probability[:, :3, :][observed]
    if np.any(selected <= 1e-300) or not np.all(np.isfinite(selected)):
        return 1e100, np.zeros_like(theta)
    ll = float(np.sum(data.counts[observed] * np.log(selected)))
    score_probability = np.zeros_like(probability)
    score_probability[:, :3, :][observed] = data.counts[observed] / selected
    survival = probability[:, :3, :3].sum(axis=2)
    alive_mask = data.alive_unknown > 0
    if np.any(survival[alive_mask] <= 1e-300):
        return 1e100, np.zeros_like(theta)
    ll += float(np.sum(data.alive_unknown[alive_mask] * np.log(survival[alive_mask])))
    alive_score = np.divide(data.alive_unknown, survival, out=np.zeros_like(data.alive_unknown), where=alive_mask)
    score_probability[:, :3, :3] += alive_score[:, :, None]
    _, score_q = legacy.batch_expm_and_adjoint(q, data.durations, score_probability)
    edge_score = np.column_stack([
        rates[:, edge] * (score_q[:, origin, destination] - score_q[:, origin, origin])
        for edge, (origin, destination) in enumerate(legacy.EDGES)
    ])
    grad_alpha = edge_score.sum(axis=0)
    grad_delta = np.vstack([edge_score[data.periods == period].sum(axis=0) for period in range(1, n_periods)])
    grad_beta = edge_score.T @ exposure
    if model == "shared":
        grad_gamma = np.zeros((3, K))
        for family in range(3):
            grad_gamma[family] = edge_score[:, legacy.EDGE_FAMILY == family].sum(axis=1) @ data.confounders
        tail = grad_gamma.ravel()
    elif model == "primary":
        grad_living = edge_score[:, :6].T @ data.confounders
        grad_death = edge_score[:, 6:].sum(axis=1) @ data.confounders
        tail = np.concatenate([grad_living.ravel(), grad_death])
    else:
        tail = (edge_score.T @ data.confounders).ravel()
    gradient = np.concatenate([grad_alpha, grad_delta.ravel(), grad_beta.ravel(), tail])
    return -ll, -gradient


def initial_from_old(domain: str, model: str, n_periods: int) -> np.ndarray:
    if model == "primary" and n_periods == 3:
        return np.load(ROOT / "initial_values" / domain / "living_specific_death_shared_parameters.npz")["theta"]
    if model == "full" and n_periods == 3:
        return np.load(ROOT / "initial_values" / domain / "unrestricted_primary_parameters.npz")["theta"]
    old = np.load(ROOT / "initial_values" / domain / "age_updated_model_parameters.npz")["theta"]
    alpha = old[:E]
    old_delta = old[E:3 * E].reshape(2, E)
    beta = old[3 * E:5 * E].reshape(E, 2)
    gamma_shared = old[5 * E:].reshape(3, K)
    if n_periods == 4:
        delta = np.vstack([np.zeros(E), old_delta])
    elif n_periods == 3:
        delta = old_delta
    else:
        delta = np.zeros((n_periods - 1, E))
    common = [alpha, delta.ravel(), beta.ravel()]
    if model == "shared":
        return np.concatenate([*common, gamma_shared.ravel()])
    if model == "primary":
        living = gamma_shared[legacy.EDGE_FAMILY[:6]]
        return np.concatenate([*common, living.ravel(), gamma_shared[2]])
    edge = gamma_shared[legacy.EDGE_FAMILY]
    return np.concatenate([*common, edge.ravel()])


def convert_initial(theta: np.ndarray, from_model: str, to_model: str, n_periods: int) -> np.ndarray:
    alpha, delta, beta, gamma = unpack(theta, from_model, n_periods)
    common = [alpha, delta.ravel(), beta.ravel()]
    if from_model == "shared" and to_model == "primary":
        living = gamma[legacy.EDGE_FAMILY[:6]]
        return np.concatenate([*common, living.ravel(), gamma[2]])
    if from_model == "primary" and to_model == "full":
        living, death = gamma
        return np.concatenate([*common, living.ravel(), np.tile(death, (3, 1)).ravel()])
    raise ValueError((from_model, to_model))


def fit_model(data: PeriodData, model: str, n_periods: int, initial: np.ndarray, maxiter: int):
    if len(initial) != parameter_count(model, n_periods):
        raise ValueError("Initial parameter length mismatch")
    common = E + (n_periods - 1) * E + 2 * E
    bounds = [(math.log(1e-6), math.log(5.0))] * E + [(-4.0, 4.0)] * (common - E)
    bounds += [(-5.0, 5.0)] * (len(initial) - common)
    scale = np.concatenate([np.full(E, 0.01), np.full(common - E, 0.01), np.full(len(initial) - common, 0.002)])
    scaled_bounds = [((lo - value) / factor, (hi - value) / factor) for (lo, hi), value, factor in zip(bounds, initial, scale)]
    def scaled_objective(increment):
        value, gradient = objective(initial + scale * increment, model, n_periods, data)
        return value, gradient * scale
    started = time.time()
    result = minimize(
        scaled_objective, np.zeros_like(initial), method="L-BFGS-B", jac=True, bounds=scaled_bounds,
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-6, "maxls": 100, "maxcor": 80},
    )
    theta = initial + scale * result.x
    nll, gradient = objective(theta, model, n_periods, data)
    active = sum(abs(value - lo) < 1e-6 or abs(value - hi) < 1e-6 for value, (lo, hi) in zip(theta, bounds))
    audit = {
        "model": model, "parameters": len(theta), "negative_log_likelihood": float(nll),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "max_abs_gradient": float(np.max(np.abs(gradient))),
        "active_bound_parameters": int(active), "elapsed_seconds": round(time.time() - started, 2),
    }
    return theta, audit


def numerical_hessian(theta: np.ndarray, data: PeriodData, n_periods: int, model: str = "primary", step: float = 2e-4) -> np.ndarray:
    hessian = np.zeros((len(theta), len(theta)))
    for index in range(len(theta)):
        plus = theta.copy(); plus[index] += step
        minus = theta.copy(); minus[index] -= step
        hessian[:, index] = (
            objective(plus, model, n_periods, data)[1] - objective(minus, model, n_periods, data)[1]
        ) / (2 * step)
    return (hessian + hessian.T) / 2


def interval_scores(theta: np.ndarray, intervals: pd.DataFrame, n_periods: int, model: str = "primary") -> np.ndarray:
    alpha, delta, beta, gamma = unpack(theta, model, n_periods)
    periods = intervals["period_code"].to_numpy(dtype=int)
    exposure_code = intervals["exposure_code"].to_numpy(dtype=int)
    exposure = np.column_stack((exposure_code == 1, exposure_code == 2)).astype(float)
    confounders = intervals[legacy.CONFOUNDER_NAMES].to_numpy(dtype=float)
    eta = alpha[None, :] + exposure @ beta.T
    nonreference = periods > 0
    eta[nonreference] += delta[periods[nonreference] - 1]
    if model == "shared":
        eta += (confounders @ gamma.T)[:, legacy.EDGE_FAMILY]
    elif model == "primary":
        living, death = gamma
        eta[:, :6] += confounders @ living.T
        eta[:, 6:] += (confounders @ death)[:, None]
    else:
        eta += confounders @ gamma.T
    rates = np.exp(np.clip(eta, -30, 8))
    n = len(intervals)
    q = np.zeros((n, 4, 4))
    for edge, (origin, destination) in enumerate(legacy.EDGES): q[:, origin, destination] = rates[:, edge]
    q[:, np.arange(3), np.arange(3)] = -q[:, :3, :].sum(axis=2)
    probability, _ = legacy.batch_expm_and_adjoint(q, intervals["interval_years"].to_numpy(dtype=float))
    rows = np.arange(n)
    origins = intervals["from_code"].to_numpy(dtype=int)
    destinations = intervals["to_code"].to_numpy(dtype=int)
    score_probability = np.zeros_like(probability)
    known = destinations != legacy.ALIVE_UNKNOWN_CODE
    selected = probability[rows[known], origins[known], destinations[known]]
    score_probability[rows[known], origins[known], destinations[known]] = 1.0 / selected
    unknown_rows = rows[~known]
    if len(unknown_rows):
        survival = probability[unknown_rows, origins[~known], :3].sum(axis=1)
        for destination in range(3):
            score_probability[unknown_rows, origins[~known], destination] = 1.0 / survival
    _, score_q = legacy.batch_expm_and_adjoint(q, intervals["interval_years"].to_numpy(dtype=float), score_probability)
    edge_score = np.column_stack([
        rates[:, edge] * (score_q[:, origin, destination] - score_q[:, origin, origin])
        for edge, (origin, destination) in enumerate(legacy.EDGES)
    ])
    score_alpha = edge_score
    score_delta = np.zeros((n, (n_periods - 1) * E))
    for period in range(1, n_periods):
        mask = periods == period
        score_delta[mask, (period - 1) * E:period * E] = edge_score[mask]
    score_beta = (edge_score[:, :, None] * exposure[:, None, :]).reshape(n, 2 * E)
    if model == "shared":
        score_gamma = np.zeros((n, 3, K))
        for family in range(3):
            score_gamma[:, family, :] = edge_score[:, legacy.EDGE_FAMILY == family].sum(axis=1)[:, None] * confounders
        tail = score_gamma.reshape(n, 3 * K)
    elif model == "primary":
        score_living = (edge_score[:, :6, None] * confounders[:, None, :]).reshape(n, 6 * K)
        score_death = edge_score[:, 6:].sum(axis=1)[:, None] * confounders
        tail = np.column_stack((score_living, score_death))
    else:
        tail = (edge_score[:, :, None] * confounders[:, None, :]).reshape(n, E * K)
    return np.column_stack((score_alpha, score_delta, score_beta, tail))


def robust_covariance(theta: np.ndarray, model_covariance: np.ndarray, intervals: pd.DataFrame, n_periods: int, model: str = "primary"):
    scores = interval_scores(theta, intervals, n_periods, model)
    frame = pd.DataFrame(scores)
    frame.insert(0, "cluster_id", intervals["cluster_id"].astype(str).to_numpy())
    clusters = frame.groupby("cluster_id", sort=False).sum(numeric_only=True).to_numpy(dtype=float)
    meat = clusters.T @ clusters
    n, g, p = len(intervals), len(clusters), len(theta)
    correction = (g / (g - 1)) * ((n - 1) / (n - p))
    covariance = model_covariance @ (correction * meat) @ model_covariance
    covariance = (covariance + covariance.T) / 2
    return covariance, {
        "intervals": n, "clusters": g, "parameters": p, "finite_sample_correction": float(correction),
        "max_abs_total_score": float(np.max(np.abs(scores.sum(axis=0)))),
    }


def hr_rows(domain: str, theta: np.ndarray, covariance: np.ndarray, n_periods: int, model: str = "primary") -> list[dict]:
    spec = legacy.DOMAIN_SPECS[domain]
    _, _, beta, _ = unpack(theta, model, n_periods)
    offset = E + (n_periods - 1) * E
    rows = []
    for edge, (origin, destination) in enumerate(legacy.EDGES):
        for exposure_index, exposure_level in enumerate(spec.exposure_labels[1:]):
            index = offset + edge * 2 + exposure_index
            estimate = float(beta[edge, exposure_index])
            se = math.sqrt(max(float(covariance[index, index]), 0))
            rows.append({
                "domain": domain, "from_state": spec.state_labels[origin], "to_state": spec.state_labels[destination],
                "contrast": f"{exposure_level} vs {spec.exposure_labels[0]}", "log_hr": estimate,
                "hr": math.exp(estimate), "robust_se_log_hr": se,
                "ci95_low": math.exp(estimate - 1.96 * se), "ci95_high": math.exp(estimate + 1.96 * se),
                "p_value": math.erfc(abs(estimate / se) / math.sqrt(2)) if se else np.nan,
            })
    return rows


def probability_rows(domain: str, theta: np.ndarray, covariance: np.ndarray, intervals: pd.DataFrame, n_periods: int, draws: int, model: str = "primary"):
    spec = legacy.DOMAIN_SPECS[domain]
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    min_before = float(values.min())
    cutoff = max(float(values.max()) * 1e-10, 1e-12)
    adjusted = int((values < cutoff).sum())
    covariance_psd = vectors @ np.diag(np.clip(values, cutoff, None)) @ vectors.T
    rng = np.random.default_rng(20260827 + (0 if domain == "pain" else 1))
    simulations = rng.multivariate_normal(theta, covariance_psd, size=draws, method="eigh")
    rows = []
    for period_index, start_wave in enumerate(PRIMARY_PERIODS):
        mean_z = intervals.loc[intervals["start_wave"].eq(start_wave), legacy.CONFOUNDER_NAMES].mean().to_numpy(dtype=float)
        for exposure_code, exposure_state in enumerate(spec.exposure_labels):
            data = PeriodData(
                durations=np.asarray([2.0]), periods=np.asarray([period_index]), exposure=np.asarray([exposure_code]),
                confounders=mean_z[None, :], counts=np.zeros((1, 3, 4)), alive_unknown=np.zeros((1, 3)),
            )
            point_eta, _ = linear_predictor(theta, model, n_periods, data)
            point_q = np.zeros((1, 4, 4))
            for edge, (origin, destination) in enumerate(legacy.EDGES): point_q[:, origin, destination] = np.exp(point_eta[:, edge])
            point_q[:, np.arange(3), np.arange(3)] = -point_q[:, :3, :].sum(axis=2)
            point_probability, _ = legacy.batch_expm_and_adjoint(point_q, np.asarray([2.0]))
            sim_data = PeriodData(
                durations=np.full(draws, 2.0), periods=np.full(draws, period_index), exposure=np.full(draws, exposure_code),
                confounders=np.tile(mean_z, (draws, 1)), counts=np.zeros((draws, 3, 4)), alive_unknown=np.zeros((draws, 3)),
            )
            sim_eta, _ = linear_predictor(simulations, "primary", n_periods, sim_data) if False else (None, None)
            # Vectorised expansion of the primary parameterisation for draws.
            alpha = simulations[:, :E]
            cursor = E
            delta = simulations[:, cursor:cursor + (n_periods - 1) * E].reshape(draws, n_periods - 1, E); cursor += (n_periods - 1) * E
            beta = simulations[:, cursor:cursor + 2 * E].reshape(draws, E, 2); cursor += 2 * E
            eta = alpha.copy()
            if period_index > 0: eta += delta[:, period_index - 1, :]
            if exposure_code > 0: eta += beta[:, :, exposure_code - 1]
            if model == "shared":
                gamma = simulations[:, cursor:].reshape(draws, 3, K)
                family_effect = np.einsum("dfk,k->df", gamma, mean_z)
                eta += family_effect[:, legacy.EDGE_FAMILY]
            elif model == "primary":
                living = simulations[:, cursor:cursor + 6 * K].reshape(draws, 6, K); cursor += 6 * K
                death = simulations[:, cursor:].reshape(draws, K)
                eta[:, :6] += np.einsum("dek,k->de", living, mean_z)
                eta[:, 6:] += (death @ mean_z)[:, None]
            else:
                gamma = simulations[:, cursor:].reshape(draws, E, K)
                eta += np.einsum("dek,k->de", gamma, mean_z)
            q = np.zeros((draws, 4, 4))
            rates = np.exp(np.clip(eta, -30, 8))
            for edge, (origin, destination) in enumerate(legacy.EDGES): q[:, origin, destination] = rates[:, edge]
            q[:, np.arange(3), np.arange(3)] = -q[:, :3, :].sum(axis=2)
            probability, _ = legacy.batch_expm_and_adjoint(q, np.full(draws, 2.0))
            for origin in range(3):
                for destination in range(4):
                    sample = probability[:, origin, destination]
                    rows.append({
                        "domain": domain, "period": PERIOD_LABELS[start_wave], "exposure_state": exposure_state,
                        "origin_state": spec.state_labels[origin], "destination_state": spec.state_labels[destination],
                        "point_estimate": float(point_probability[0, origin, destination]),
                        "simulation_median": float(np.median(sample)), "ci95_low": float(np.quantile(sample, 0.025)),
                        "ci95_high": float(np.quantile(sample, 0.975)), "draws": draws,
                    })
    return rows, {"minimum_eigenvalue_before": min_before, "eigenvalues_adjusted": adjusted, "cutoff": cutoff}


def extract_severe(domain: str, theta: np.ndarray, n_periods: int, model: str = "primary") -> dict:
    _, _, beta, _ = unpack(theta, model, n_periods)
    edge = 1
    return {
        "domain": domain,
        "pathway": "P0->P2, F2 vs F0" if domain == "pain" else "F0->F2, P2 vs P0",
        "hr": float(np.exp(beta[edge, 1])),
    }


def run_primary_domain(domain: str, maxiter: int, draws: int) -> dict:
    out = FIVE / "models" / domain
    out.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(FIVE / "cohort" / "CHARLS_分析长格式_45岁及以上.csv", dtype={"person_id": "string", "household_id": "string"}, low_memory=False)
    intervals, interval_audit = prepare_intervals(domain, long)
    intervals.to_csv(out / "fivewave_primary_intervals.csv", index=False)
    data = group_intervals(intervals)
    shared0 = initial_from_old(domain, "shared", 4)
    shared, shared_audit = fit_model(data, "shared", 4, shared0, maxiter)
    primary0 = convert_initial(shared, "shared", "primary", 4)
    primary, primary_audit = fit_model(data, "primary", 4, primary0, maxiter)
    full0 = convert_initial(primary, "primary", "full", 4)
    full, full_audit = fit_model(data, "full", 4, full0, maxiter)
    hessian = numerical_hessian(primary, data, 4)
    model_covariance = np.linalg.pinv(hessian, rcond=1e-10)
    model_covariance = (model_covariance + model_covariance.T) / 2
    robust, robust_audit = robust_covariance(primary, model_covariance, intervals, 4)
    hrs = hr_rows(domain, primary, robust, 4)
    probabilities, probability_audit = probability_rows(domain, primary, robust, intervals, 4, draws)
    np.savez_compressed(out / "fivewave_nested_parameters.npz", shared=shared, primary=primary, full=full,
                        hessian=hessian, model_covariance=model_covariance, household_robust_covariance=robust)
    pd.DataFrame(hrs).to_csv(out / "fivewave_primary_household_robust_hr.csv", index=False)
    pd.DataFrame(probabilities).to_csv(out / "fivewave_primary_period_probabilities.csv", index=False)
    comparisons = {
        "shared_vs_primary": {"chi_square": float(2 * (shared_audit["negative_log_likelihood"] - primary_audit["negative_log_likelihood"])), "df": 32},
        "primary_vs_full": {"chi_square": float(2 * (primary_audit["negative_log_likelihood"] - full_audit["negative_log_likelihood"])), "df": 16},
    }
    for item in comparisons.values(): item["p_value"] = float(chi2.sf(item["chi_square"], item["df"]))
    audit = {
        "domain": domain, "people": int(intervals.person_id.nunique()), "intervals": int(len(intervals)),
        "household_clusters": int(intervals.cluster_id.nunique()),
        "period_interval_counts": {str(k): int(v) for k, v in intervals.start_wave.value_counts().sort_index().items()},
        "entry_wave_counts": interval_audit.get("entry_wave_counts"), "interval_build": interval_audit,
        "shared_fit": shared_audit, "primary_fit": primary_audit, "full_fit": full_audit, "comparisons": comparisons,
        "hessian_minimum_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
        "hessian_condition_number": float(np.linalg.cond(hessian)), "robust_covariance": robust_audit,
        "probability_covariance": probability_audit, "probability_draws": draws,
    }
    (out / "fivewave_primary_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"audit": audit, "theta": primary, "intervals": intervals, "long": long}


def run_sensitivities(domain: str, primary_result: dict, maxiter: int) -> list[dict]:
    out = FIVE / "models" / domain
    primary = primary_result["theta"]
    long = primary_result["long"]
    rows = []
    # Cross-sectional survey-weighted likelihood, complete weights only.
    intervals = primary_result["intervals"]
    weights, weight_audit = sampling_weights(intervals)
    weighted_data = group_intervals(intervals, weights)
    weighted, weighted_fit = fit_model(weighted_data, "primary", 4, primary, maxiter)
    rows.append({"analysis": "cross-sectional-weighted", **extract_severe(domain, weighted, 4), **weighted_fit})
    weight_audit["fit"] = weighted_fit
    (out / "fivewave_weighted_audit.json").write_text(json.dumps(weight_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Strict complete 6-BADL + 5-IADL function definition.
    strict_intervals, strict_audit = prepare_intervals(domain, long, strict_function=True)
    strict_data = group_intervals(strict_intervals)
    strict, strict_fit = fit_model(strict_data, "primary", 4, primary, maxiter)
    rows.append({"analysis": "strict-complete-11-item-function", **extract_severe(domain, strict, 4),
                 "people": int(strict_intervals.person_id.nunique()), "intervals": int(len(strict_intervals)), **strict_fit})
    (out / "fivewave_strict_function_audit.json").write_text(json.dumps({"interval_build": strict_audit, "fit": strict_fit}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Restrict 2013 death confirmation to direct Exit Interview records.
    exit_ids = pd.read_stata(FIVE / "stage" / "2013" / "Exit_Interview.dta", columns=["ID"], convert_categoricals=False)["ID"].astype("string").str.replace(r"\.0$", "", regex=True)
    exit_only = long.copy()
    wave2013 = exit_only.wave.eq(2013)
    not_exit = wave2013 & ~exit_only.person_id.astype("string").isin(set(exit_ids.dropna()))
    weight_only_death = not_exit & exit_only.death_confirmed.eq(1)
    exit_only.loc[weight_only_death, "death_confirmed"] = 0
    exit_only.loc[weight_only_death, "died_raw"] = 0
    for col in ["pain_state_with_death", "function_state", "function_state_complete11", "joint_state"]:
        exit_only.loc[weight_only_death, col] = pd.NA
    death_intervals, death_audit = prepare_intervals(domain, exit_only)
    death_data = group_intervals(death_intervals)
    death_fit_theta, death_fit = fit_model(death_data, "primary", 4, primary, maxiter)
    rows.append({"analysis": "2013-exit-interview-deaths-only", **extract_severe(domain, death_fit_theta, 4),
                 "weight_only_deaths_reclassified": int(weight_only_death.sum()), **death_fit})
    (out / "fivewave_exit_death_audit.json").write_text(json.dumps({"interval_build": death_audit, "fit": death_fit, "reclassified": int(weight_only_death.sum())}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Leave-2013-out analysis using the same model family and original three periods.
    no2013 = long.loc[long.wave.ne(2013)].copy()
    old_pairs = ((2011, 2015), (2015, 2018), (2018, 2020))
    leave_intervals, leave_audit = prepare_intervals(domain, no2013, (2011, 2015, 2018), old_pairs)
    leave_data = group_intervals(leave_intervals)
    leave0 = initial_from_old(domain, "primary", 3)
    leave, leave_fit = fit_model(leave_data, "primary", 3, leave0, maxiter)
    rows.append({"analysis": "leave-2013-out", **extract_severe(domain, leave, 3),
                 "people": int(leave_intervals.person_id.nunique()), "intervals": int(len(leave_intervals)), **leave_fit})
    (out / "leave_2013_out_audit.json").write_text(json.dumps({"interval_build": leave_audit, "fit": leave_fit}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "fivewave_sensitivity_summary.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["pain", "function"], required=True)
    parser.add_argument("--maxiter", type=int, default=900)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--skip-sensitivities", action="store_true")
    args = parser.parse_args()
    result = run_primary_domain(args.domain, args.maxiter, args.draws)
    sensitivities = [] if args.skip_sensitivities else run_sensitivities(args.domain, result, args.maxiter)
    print(json.dumps({"primary": result["audit"], "sensitivities": sensitivities}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
