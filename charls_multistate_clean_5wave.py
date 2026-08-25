#!/usr/bin/env python3
"""Build harmonized CHARLS 2011/2013/2015/2018/2020 multistate analysis data.

This program implements the rules documented in the cross-wave variable dictionary:

* Pain: P0 = 0 musculoskeletal sites, P1 = 1, P2 = 2 or more.
* Function: F0 = no BADL/IADL limitation, F1 = IADL-only limitation,
  F2 = at least one BADL limitation, D = confirmed death.
* Death comes from Sample_Infor.dta `died`; nonresponse is never coded as alive.
* The 2011 person ID is harmonized by inserting a zero before its final two digits.
* In 2011/2013/2015/2018, all-six-BADL missing values are recoded to zero only when
  the later IADL block contains at least one valid response. This identifies the
  questionnaire's structural skip over DB010-DB015 for clearly high-functioning
  respondents. Other item missingness is retained.

The script intentionally does not impose an age restriction because no age rule was
specified for this cleaning stage. It retains the CHARLS wave universes and exposes
interview/attrition flags for later analytic cohort selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = PROJECT_ROOT / "stage"
DEFAULT_OUTPUT = PROJECT_ROOT
WAVES = (2011, 2013, 2015, 2018, 2020)
WAVE_PAIRS = ((2011, 2013), (2013, 2015), (2015, 2018), (2018, 2020))

MSK_SITES = {
    2: "shoulder",
    3: "arm",
    4: "wrist",
    5: "fingers",
    8: "back",
    9: "waist",
    10: "buttocks",
    11: "leg",
    12: "knee",
    13: "ankle",
    14: "toes",
    15: "neck",
}

BADL_NAMES = (
    "dressing",
    "bathing",
    "eating",
    "bed_transfer",
    "toileting",
    "continence",
)
IADL_NAMES = (
    "household_chores",
    "hot_meals",
    "shopping",
    "medications",
    "money",
)

HEALTH_FILES = {
    2011: "health_status_and_functioning.dta",
    2013: "Health_Status_and_Functioning.dta",
    2015: "Health_Status_and_Functioning.dta",
    2018: "Health_Status_and_Functioning.dta",
    2020: "Health_Status_and_Functioning.dta",
}
SAMPLE_FILES = {year: "Sample_Infor.dta" for year in (2013, 2015, 2018, 2020)}
WEIGHT_FILES = {
    2011: "weight.dta",
    2013: "Weights.dta",
    2015: "Weights.dta",
    2018: "Weights.dta",
    2020: "Weights.dta",
}

PAIN_SCREEN = {2011: "da041", 2013: "wb16", 2015: "da041", 2018: "da041_w4", 2020: "da027"}
PAIN_SITE_PREFIX = {2011: "da042s", 2013: "da042s", 2015: "da042s", 2018: "da042_s", 2020: "da028_s"}
BADL_RAW = {
    2011: [f"db{i:03d}" for i in range(10, 16)],
    2013: [f"db{i:03d}" for i in range(10, 16)],
    2015: [f"db{i:03d}" for i in range(10, 16)],
    2018: [f"db{i:03d}" for i in range(10, 16)],
    2020: [f"db{i:03d}" for i in (1, 3, 5, 7, 9, 11)],
}
IADL_RAW = {
    2011: ["db016", "db017", "db018", "db020", "db019"],
    2013: ["db016", "db017", "db018", "db020", "db019"],
    2015: ["db016", "db017", "db018", "db020", "db019"],
    2018: ["db016", "db017", "db018", "db020", "db019"],
    2020: ["db012", "db014", "db016", "db020", "db022"],
}
PHONE_RAW = {2013: "db035", 2015: "db035", 2018: "db035", 2020: "db018"}

PAIN_STATES = ("P0", "P1", "P2", "D")
FUNCTION_STATES = ("F0", "F1", "F2", "D")
JOINT_STATES = tuple(f"P{p}F{f}" for p in range(3) for f in range(3)) + ("D",)


@dataclass
class SourceAudit:
    wave: int
    source: str
    file: str
    rows_raw: int
    unique_raw_id: int
    duplicate_raw_id_rows: int
    invalid_standard_id_rows: int
    rows_retained: int
    duplicate_standard_id_rows: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE, help="Folder containing wave subfolders")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder")
    return parser.parse_args()


def clean_string(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def standardize_person_id(raw: pd.Series, wave: int) -> pd.Series:
    value = clean_string(raw)
    digits = value.str.fullmatch(r"\d+").fillna(False)
    if wave == 2011:
        valid = digits & value.str.len().eq(11)
        result = value.str.slice(0, -2) + "0" + value.str.slice(-2)
    else:
        valid = digits & value.str.len().eq(12)
        result = value
    return result.where(valid).astype("string")


def standardize_household_id(raw: pd.Series, wave: int) -> pd.Series:
    value = clean_string(raw)
    digits = value.str.fullmatch(r"\d+").fillna(False)
    if wave == 2011:
        valid = digits & value.str.len().eq(9)
        result = value + "0"
    else:
        valid = digits & value.str.len().eq(10)
        result = value
    return result.where(valid).astype("string")


def standardize_community_id(raw: pd.Series) -> pd.Series:
    value = clean_string(raw)
    valid = value.str.fullmatch(r"\d{7}").fillna(False)
    return value.where(valid).astype("string")


def read_stata_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    reader = pd.io.stata.StataReader(path, convert_categoricals=False)
    available = set(reader.variable_labels())
    requested = list(dict.fromkeys(columns))
    missing = sorted(set(requested) - available)
    if missing:
        raise KeyError(f"Missing variables in {path}: {missing}")
    return pd.read_stata(path, columns=requested, convert_categoricals=False)


def load_source(
    path: Path,
    wave: int,
    source: str,
    variables: Iterable[str],
) -> tuple[pd.DataFrame, SourceAudit, pd.DataFrame]:
    columns = ["ID", "householdID", "communityID", *variables]
    frame = read_stata_columns(path, columns)
    raw_id = clean_string(frame["ID"])
    frame["person_id"] = standardize_person_id(frame["ID"], wave)
    frame[f"person_id_raw_{source}"] = raw_id
    frame[f"household_id_{source}"] = standardize_household_id(frame["householdID"], wave)
    frame[f"community_id_{source}"] = standardize_community_id(frame["communityID"])

    invalid = frame["person_id"].isna()
    invalid_details = pd.DataFrame(
        {
            "wave": wave,
            "source": source,
            "file": path.name,
            "person_id_raw": raw_id[invalid],
            "household_id_raw": clean_string(frame.loc[invalid, "householdID"]),
            "community_id_raw": clean_string(frame.loc[invalid, "communityID"]),
            "reason": "person ID is not in the expected wave-specific format",
        }
    )

    duplicate_raw = int(raw_id.duplicated(keep=False).sum())
    retained = frame.loc[~invalid].copy()
    duplicate_standard = int(retained["person_id"].duplicated(keep=False).sum())
    audit = SourceAudit(
        wave=wave,
        source=source,
        file=str(path),
        rows_raw=len(frame),
        unique_raw_id=int(raw_id.nunique(dropna=True)),
        duplicate_raw_id_rows=duplicate_raw,
        invalid_standard_id_rows=int(invalid.sum()),
        rows_retained=len(retained),
        duplicate_standard_id_rows=duplicate_standard,
    )
    if duplicate_standard:
        examples = retained.loc[retained["person_id"].duplicated(keep=False), "person_id"].head().tolist()
        raise ValueError(f"Duplicate standardized IDs in {path}: {examples}")

    retained = retained.drop(columns=["ID", "householdID", "communityID"])
    retained[f"{source}_record_present"] = 1
    return retained, audit, invalid_details


def nullable_binary_from_codes(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result.loc[numeric.eq(1)] = 0
    result.loc[numeric.isin([2, 3, 4])] = 1
    return result


def int_sum_if_complete(frame: pd.DataFrame, required: int) -> pd.Series:
    result = frame.sum(axis=1, min_count=required)
    return result.astype("Int64")


def numeric_invalid_count(series: pd.Series, valid_values: set[int | float]) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    return int((series.notna() & ~numeric.isin(valid_values)).sum())


def clean_health(
    frame: pd.DataFrame,
    wave: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    pain_audit: list[dict[str, Any]] = []
    function_audit: list[dict[str, Any]] = []

    screen_raw_name = PAIN_SCREEN[wave]
    screen_numeric = pd.to_numeric(frame[screen_raw_name], errors="coerce")
    pain_present = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if wave in (2011, 2015):
        pain_present.loc[screen_numeric.eq(1)] = 1
        pain_present.loc[screen_numeric.eq(2)] = 0
        screen_valid_values = {1, 2}
    else:
        pain_present.loc[screen_numeric.eq(1)] = 0
        pain_present.loc[screen_numeric.isin([2, 3, 4, 5])] = 1
        screen_valid_values = {1, 2, 3, 4, 5}

    site_prefix = PAIN_SITE_PREFIX[wave]
    max_site = 15 if wave in (2011, 2013) else 16
    all_site_raw_names = [f"{site_prefix}{number}" for number in range(1, max_site + 1)]
    site_numeric = frame[all_site_raw_names].apply(pd.to_numeric, errors="coerce")
    invalid_site = pd.DataFrame(False, index=frame.index, columns=all_site_raw_names)
    selected_site = pd.DataFrame(False, index=frame.index, columns=all_site_raw_names)
    for number, raw_name in enumerate(all_site_raw_names, start=1):
        selected_site[raw_name] = site_numeric[raw_name].eq(number)
        allowed = {number} if wave in (2011, 2013, 2015) else {0, number}
        invalid_site[raw_name] = frame[raw_name].notna() & ~site_numeric[raw_name].isin(allowed)
        pain_audit.append(
            {
                "wave": wave,
                "variable": raw_name,
                "check": "invalid pain-site raw code",
                "invalid_count": int(invalid_site[raw_name].sum()),
                "valid_codes": ",".join(str(value) for value in sorted(allowed)),
            }
        )

    any_invalid_site = invalid_site.any(axis=1)
    if wave in (2011, 2013, 2015):
        section_complete = selected_site.any(axis=1) & ~any_invalid_site
    else:
        section_complete = site_numeric.notna().all(axis=1) & ~any_invalid_site

    pain_section_valid = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    pain_section_valid.loc[pain_present.eq(1)] = section_complete.loc[pain_present.eq(1)].astype(int)

    site_outputs: dict[str, pd.Series] = {}
    for number, site_name in MSK_SITES.items():
        raw_name = f"{site_prefix}{number}"
        indicator = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        indicator.loc[pain_present.eq(0)] = 0
        valid_pain = pain_present.eq(1) & section_complete
        indicator.loc[valid_pain] = site_numeric.loc[valid_pain, raw_name].eq(number).astype(int)
        site_outputs[f"pain_site_{site_name}"] = indicator

    site_frame = pd.DataFrame(site_outputs)
    msk_pain_n = int_sum_if_complete(site_frame, len(MSK_SITES))
    pain_state = pd.Series(pd.NA, index=frame.index, dtype="string")
    pain_state.loc[msk_pain_n.eq(0)] = "P0"
    pain_state.loc[msk_pain_n.eq(1)] = "P1"
    pain_state.loc[msk_pain_n.ge(2)] = "P2"
    pain_state_num = msk_pain_n.clip(upper=2).astype("Int64")

    badl_raw = frame[BADL_RAW[wave]]
    iadl_raw = frame[IADL_RAW[wave]]
    badl_items = pd.DataFrame(
        {
            f"badl_{name}": nullable_binary_from_codes(badl_raw[raw])
            for name, raw in zip(BADL_NAMES, BADL_RAW[wave])
        }
    )
    iadl_items = pd.DataFrame(
        {
            f"iadl_{name}": nullable_binary_from_codes(iadl_raw[raw])
            for name, raw in zip(IADL_NAMES, IADL_RAW[wave])
        }
    )

    # Strict complete-case sensitivity status must be determined from the raw
    # 11 BADL/IADL item responses, before questionnaire structural skips are
    # recoded for the primary function-state definition.
    all11_complete_raw = (
        badl_items.notna().all(axis=1)
        & iadl_items.notna().all(axis=1)
    )

    iadl_has_valid_response = iadl_raw.apply(pd.to_numeric, errors="coerce").isin([1, 2, 3, 4]).any(axis=1)
    structural_skip = pd.Series(False, index=frame.index)
    if wave < 2020:
        structural_skip = badl_raw.isna().all(axis=1) & iadl_has_valid_response
        badl_items.loc[structural_skip, :] = 0

    badl_valid_n = badl_items.notna().sum(axis=1).astype("Int64")
    iadl_valid_n = iadl_items.notna().sum(axis=1).astype("Int64")
    badl_n = int_sum_if_complete(badl_items, len(BADL_NAMES))
    iadl5_n = int_sum_if_complete(iadl_items, len(IADL_NAMES))
    function_state_alive = pd.Series(pd.NA, index=frame.index, dtype="string")
    function_state_alive.loc[badl_n.ge(1)] = "F2"
    function_state_alive.loc[badl_n.eq(0) & iadl5_n.ge(1)] = "F1"
    function_state_alive.loc[badl_n.eq(0) & iadl5_n.eq(0)] = "F0"

    function_state_complete11 = pd.Series(pd.NA, index=frame.index, dtype="string")
    function_state_complete11.loc[all11_complete_raw & badl_n.ge(1)] = "F2"
    function_state_complete11.loc[all11_complete_raw & badl_n.eq(0) & iadl5_n.ge(1)] = "F1"
    function_state_complete11.loc[all11_complete_raw & badl_n.eq(0) & iadl5_n.eq(0)] = "F0"

    for raw, standard in zip(BADL_RAW[wave], badl_items.columns):
        function_audit.append(
            {
                "wave": wave,
                "domain": "BADL",
                "raw_variable": raw,
                "standard_variable": standard,
                "invalid_nonmissing_count": numeric_invalid_count(frame[raw], {1, 2, 3, 4}),
            }
        )
    for raw, standard in zip(IADL_RAW[wave], iadl_items.columns):
        function_audit.append(
            {
                "wave": wave,
                "domain": "IADL",
                "raw_variable": raw,
                "standard_variable": standard,
                "invalid_nonmissing_count": numeric_invalid_count(frame[raw], {1, 2, 3, 4}),
            }
        )

    meta_cols = [
        "person_id",
        "person_id_raw_health",
        "household_id_health",
        "community_id_health",
        "health_record_present",
    ]
    output = frame[meta_cols].copy()
    output["pain_screen_raw"] = screen_numeric
    output["pain_present"] = pain_present
    output["pain_site_section_valid"] = pain_section_valid
    output = pd.concat([output, site_frame], axis=1)
    output["msk_pain_n"] = msk_pain_n
    output["pain_state_alive"] = pain_state
    output["pain_state_num_alive"] = pain_state_num
    output = pd.concat([output, badl_items, iadl_items], axis=1)
    output["badl_structural_skip"] = structural_skip.astype("Int64")
    output["badl_valid_n"] = badl_valid_n
    output["badl_n"] = badl_n
    output["iadl5_valid_n"] = iadl_valid_n
    output["iadl5_n"] = iadl5_n
    output["function_state_alive"] = function_state_alive
    output["all11_complete_raw"] = all11_complete_raw.astype("Int64")
    output["function_state_complete11_alive"] = function_state_complete11

    if wave in PHONE_RAW:
        output["iadl_phone"] = nullable_binary_from_codes(frame[PHONE_RAW[wave]])
    else:
        output["iadl_phone"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    pain_audit.insert(
        0,
        {
            "wave": wave,
            "variable": screen_raw_name,
            "check": "invalid pain-screen raw code",
            "invalid_count": numeric_invalid_count(frame[screen_raw_name], screen_valid_values),
            "valid_codes": ",".join(str(value) for value in sorted(screen_valid_values)),
        },
    )
    return output, pain_audit, function_audit


def coalesce_columns(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    available = [name for name in names if name in frame]
    if not available:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    result = frame[available[0]].copy()
    for name in available[1:]:
        result = result.combine_first(frame[name])
    return result


def nullable_int(series: pd.Series, valid_min: int | None = None, valid_max: int | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.mod(1).eq(0))
    if valid_min is not None:
        numeric = numeric.where(numeric.ge(valid_min))
    if valid_max is not None:
        numeric = numeric.where(numeric.le(valid_max))
    return numeric.round().astype("Int64")


def build_wave(
    stage: Path,
    wave: int,
) -> tuple[pd.DataFrame, list[SourceAudit], list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    audits: list[SourceAudit] = []
    invalid_details: list[pd.DataFrame] = []

    pain_site_count = 15 if wave in (2011, 2013) else 16
    health_variables = [PAIN_SCREEN[wave]]
    health_variables += [f"{PAIN_SITE_PREFIX[wave]}{number}" for number in range(1, pain_site_count + 1)]
    health_variables += BADL_RAW[wave] + IADL_RAW[wave]
    if wave in PHONE_RAW:
        health_variables.append(PHONE_RAW[wave])
    health_path = stage / str(wave) / HEALTH_FILES[wave]
    health_raw, health_audit, health_invalid = load_source(
        health_path, wave, "health", health_variables
    )
    audits.append(health_audit)
    invalid_details.append(health_invalid)
    health, pain_audit, function_audit = clean_health(health_raw, wave)

    sources = [health]
    if wave in SAMPLE_FILES:
        sample_path = stage / str(wave) / SAMPLE_FILES[wave]
        sample, sample_audit, sample_invalid = load_source(
            sample_path, wave, "sample", ["iyear", "imonth", "died"]
        )
        sample = sample.rename(
            columns={"iyear": "interview_year_raw", "imonth": "interview_month_raw", "died": "died_raw"}
        )
        audits.append(sample_audit)
        invalid_details.append(sample_invalid)
        sources.append(sample)

    weight_path = stage / str(wave) / WEIGHT_FILES[wave]
    weight_var = "ind_weight_ad2" if wave == 2011 else "INDV_weight_ad2"
    weight_variables = [weight_var]
    if wave == 2011:
        weight_variables += ["iyear", "imonth"]
    weight, weight_audit, weight_invalid = load_source(
        weight_path, wave, "weight", weight_variables
    )
    rename_weight = {weight_var: "individual_weight"}
    if wave == 2011:
        rename_weight.update({"iyear": "interview_year_raw", "imonth": "interview_month_raw"})
    weight = weight.rename(columns=rename_weight)
    audits.append(weight_audit)
    invalid_details.append(weight_invalid)
    sources.append(weight)

    if wave == 2020:
        exit_path = stage / "2020" / "Exit_Module.dta"
        exit_frame, exit_audit, exit_invalid = load_source(
            exit_path, wave, "exit", ["exb001_1", "exb001_2", "exb001_3"]
        )
        exit_frame = exit_frame.rename(
            columns={"exb001_1": "death_year", "exb001_2": "death_month", "exb001_3": "death_day"}
        )
        audits.append(exit_audit)
        invalid_details.append(exit_invalid)
        sources.append(exit_frame)

    all_ids = pd.concat([source[["person_id"]] for source in sources], ignore_index=True).drop_duplicates()
    wave_frame = all_ids
    for source in sources:
        wave_frame = wave_frame.merge(source, on="person_id", how="left", validate="one_to_one")

    source_names = ("health", "sample", "weight", "exit")
    wave_frame["person_id_raw"] = coalesce_columns(
        wave_frame, [f"person_id_raw_{name}" for name in source_names]
    )
    wave_frame["household_id"] = coalesce_columns(
        wave_frame, [f"household_id_{name}" for name in source_names]
    )
    wave_frame["community_id"] = coalesce_columns(
        wave_frame, [f"community_id_{name}" for name in source_names]
    )
    drop_meta = [
        column
        for name in source_names
        for column in (f"person_id_raw_{name}", f"household_id_{name}", f"community_id_{name}")
        if column in wave_frame
    ]
    wave_frame = wave_frame.drop(columns=drop_meta)
    for source in source_names:
        flag = f"{source}_record_present"
        if flag not in wave_frame:
            wave_frame[flag] = 0
        wave_frame[flag] = wave_frame[flag].fillna(0).astype("Int64")

    wave_frame["wave"] = wave
    interview_year_raw = wave_frame.get(
        "interview_year_raw", pd.Series(pd.NA, index=wave_frame.index)
    )
    interview_month_raw = wave_frame.get(
        "interview_month_raw", pd.Series(pd.NA, index=wave_frame.index)
    )
    wave_frame["interview_year"] = nullable_int(interview_year_raw, 2000, 2030)
    wave_frame["interview_month"] = nullable_int(interview_month_raw, 1, 12)
    wave_frame["interview_year_invalid_code"] = (
        interview_year_raw.notna() & wave_frame["interview_year"].isna()
    ).astype("Int64")
    wave_frame["interview_month_invalid_code"] = (
        interview_month_raw.notna() & wave_frame["interview_month"].isna()
    ).astype("Int64")
    date_parts = pd.DataFrame(
        {
            "year": wave_frame["interview_year"],
            "month": wave_frame["interview_month"],
            "day": 15,
        }
    )
    wave_frame["interview_date"] = pd.to_datetime(date_parts, errors="coerce")
    weight_raw = wave_frame.get("individual_weight", pd.Series(np.nan, index=wave_frame.index))
    weight_numeric = pd.to_numeric(weight_raw, errors="coerce")
    wave_frame["weight_invalid_or_nonpositive"] = (
        weight_raw.notna() & (weight_numeric.isna() | weight_numeric.le(0))
    ).astype("Int64")
    wave_frame["individual_weight"] = weight_numeric.where(weight_numeric.gt(0))

    died_source = wave_frame.get("died_raw", pd.Series(np.nan, index=wave_frame.index))
    died_numeric = pd.to_numeric(died_source, errors="coerce")
    wave_frame["died_invalid_code"] = (
        died_source.notna() & ~died_numeric.isin([0, 1])
    ).astype("Int64")
    wave_frame["died_raw"] = died_numeric.where(died_numeric.isin([0, 1])).astype("Int64")
    wave_frame["death_confirmed"] = died_numeric.eq(1).astype("Int64")
    health_present = wave_frame["health_record_present"].eq(1)
    death = wave_frame["death_confirmed"].eq(1)
    vital = pd.Series("no_health_no_death", index=wave_frame.index, dtype="string")
    vital.loc[health_present & ~death] = "interviewed_alive"
    vital.loc[death & ~health_present] = "death_confirmed"
    vital.loc[death & health_present] = "death_health_conflict"
    wave_frame["vital_observation"] = vital
    wave_frame["death_state"] = pd.Series(pd.NA, index=wave_frame.index, dtype="string")
    wave_frame.loc[death, "death_state"] = "D"

    wave_frame["pain_state"] = wave_frame.get("pain_state_alive", pd.Series(pd.NA, index=wave_frame.index)).astype("string")
    wave_frame.loc[death, "pain_state"] = pd.NA
    wave_frame["pain_state_num"] = wave_frame.get(
        "pain_state_num_alive", pd.Series(pd.NA, index=wave_frame.index)
    ).astype("Int64")
    wave_frame.loc[death, "pain_state_num"] = pd.NA
    wave_frame["pain_state_with_death"] = wave_frame["pain_state"]
    wave_frame.loc[death, "pain_state_with_death"] = "D"

    wave_frame["function_state"] = wave_frame.get(
        "function_state_alive", pd.Series(pd.NA, index=wave_frame.index)
    ).astype("string")
    wave_frame.loc[death, "function_state"] = "D"
    wave_frame["function_state_complete11"] = wave_frame.get(
        "function_state_complete11_alive", pd.Series(pd.NA, index=wave_frame.index)
    ).astype("string")
    wave_frame.loc[death, "function_state_complete11"] = "D"
    function_num_map = {"F0": 0, "F1": 1, "F2": 2, "D": 3}
    wave_frame["function_state_num"] = wave_frame["function_state"].map(function_num_map).astype("Int64")

    joint = pd.Series(pd.NA, index=wave_frame.index, dtype="string")
    active_joint = wave_frame["pain_state"].notna() & wave_frame["function_state"].isin(["F0", "F1", "F2"])
    joint.loc[active_joint] = (
        wave_frame.loc[active_joint, "pain_state"].str.slice(0, 2)
        + wave_frame.loc[active_joint, "function_state"].str.slice(0, 2)
    )
    joint.loc[death] = "D"
    wave_frame["joint_state"] = joint

    wave_frame["health_death_conflict"] = (health_present & death).astype("Int64")
    return wave_frame, audits, invalid_details, pain_audit, function_audit


def first_nonmissing(series: pd.Series) -> Any:
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


def build_transition_outputs(
    long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    state_columns = {
        "pain": "pain_state_with_death",
        "function": "function_state",
        "joint": "joint_state",
    }
    for wave_from, wave_to in WAVE_PAIRS:
        left_cols = [
            "person_id",
            "wave",
            "interview_date",
            "vital_observation",
            "health_record_present",
            "post_death_record",
            *state_columns.values(),
        ]
        right_cols = left_cols.copy()
        left = long.loc[long["wave"].eq(wave_from), left_cols].copy()
        right = long.loc[long["wave"].eq(wave_to), right_cols].copy()
        left = left.rename(columns={column: f"{column}_from" for column in left_cols if column != "person_id"})
        right = right.rename(columns={column: f"{column}_to" for column in right_cols if column != "person_id"})
        pair = left.merge(right, on="person_id", how="left", validate="one_to_one", indicator="next_record_merge")
        pair["wave_pair"] = f"{wave_from}-{wave_to}"
        from_date = pd.to_datetime(pair["interview_date_from"], errors="coerce")
        to_date = pd.to_datetime(pair["interview_date_to"], errors="coerce")
        pair["interval_years"] = (to_date - from_date).dt.days / 365.25

        for domain, state_col in state_columns.items():
            origin = pair[f"{state_col}_from"].astype("string")
            destination = pair[f"{state_col}_to"].astype("string")
            eligible = origin.notna() & origin.ne("D") & pair["post_death_record_from"].fillna(0).eq(0)
            disposition = pd.Series(pd.NA, index=pair.index, dtype="string")
            observed = eligible & destination.notna() & pair["post_death_record_to"].fillna(0).eq(0)
            disposition.loc[observed] = destination.loc[observed]
            no_next = eligible & pair["next_record_merge"].eq("left_only")
            disposition.loc[no_next] = "CENSOR:no_next_record"
            next_present = eligible & pair["next_record_merge"].eq("both") & ~observed
            no_health = next_present & pair["vital_observation_to"].eq("no_health_no_death")
            disposition.loc[no_health] = "CENSOR:next_no_health_no_death"
            state_missing = next_present & pair["health_record_present_to"].fillna(0).eq(1) & disposition.isna()
            disposition.loc[state_missing] = "CENSOR:next_state_missing"
            disposition.loc[next_present & disposition.isna()] = "CENSOR:next_unknown"
            pair[f"{domain}_from_state"] = origin
            pair[f"{domain}_to_state"] = destination.where(observed)
            pair[f"{domain}_disposition"] = disposition
            pair[f"{domain}_origin_eligible"] = eligible.astype("Int64")

        keep_any = pair[[f"{domain}_origin_eligible" for domain in state_columns]].eq(1).any(axis=1)
        records.append(pair.loc[keep_any].copy())

    transition_records = pd.concat(records, ignore_index=True)
    tidy_rows: list[dict[str, Any]] = []
    censor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    state_order = {"pain": PAIN_STATES, "function": FUNCTION_STATES, "joint": JOINT_STATES}
    for domain, states in state_order.items():
        active_states = [state for state in states if state != "D"]
        for pair_label in [f"{a}-{b}" for a, b in WAVE_PAIRS]:
            subset = transition_records.loc[
                transition_records["wave_pair"].eq(pair_label)
                & transition_records[f"{domain}_origin_eligible"].eq(1)
            ].copy()
            for from_state in active_states:
                origin_rows = subset.loc[subset[f"{domain}_from_state"].eq(from_state)]
                total_origin = len(origin_rows)
                observed_rows = origin_rows.loc[origin_rows[f"{domain}_to_state"].notna()]
                total_observed = len(observed_rows)
                for to_state in states:
                    count = int(observed_rows[f"{domain}_to_state"].eq(to_state).sum())
                    tidy_rows.append(
                        {
                            "domain": domain,
                            "wave_pair": pair_label,
                            "from_state": from_state,
                            "to_state": to_state,
                            "n": count,
                            "row_pct_observed": count / total_observed if total_observed else np.nan,
                            "pct_of_all_origins": count / total_origin if total_origin else np.nan,
                        }
                    )
                dispositions = origin_rows[f"{domain}_disposition"].value_counts(dropna=False)
                for disposition, count in dispositions.items():
                    censor_rows.append(
                        {
                            "domain": domain,
                            "wave_pair": pair_label,
                            "from_state": from_state,
                            "disposition": "<missing>" if pd.isna(disposition) else str(disposition),
                            "n": int(count),
                            "pct_of_origins": int(count) / total_origin if total_origin else np.nan,
                        }
                    )

            observed_domain = subset.loc[subset[f"{domain}_to_state"].notna()]
            intervals = observed_domain["interval_years"].dropna()
            summary_rows.append(
                {
                    "domain": domain,
                    "wave_pair": pair_label,
                    "eligible_active_origins": len(subset),
                    "observed_endpoints": len(observed_domain),
                    "to_death": int(observed_domain[f"{domain}_to_state"].eq("D").sum()),
                    "censored_no_next_record": int(
                        subset[f"{domain}_disposition"].eq("CENSOR:no_next_record").sum()
                    ),
                    "censored_next_no_health_no_death": int(
                        subset[f"{domain}_disposition"].eq("CENSOR:next_no_health_no_death").sum()
                    ),
                    "censored_next_state_missing": int(
                        subset[f"{domain}_disposition"].eq("CENSOR:next_state_missing").sum()
                    ),
                    "interval_years_median": float(intervals.median()) if len(intervals) else np.nan,
                    "interval_years_min": float(intervals.min()) if len(intervals) else np.nan,
                    "interval_years_max": float(intervals.max()) if len(intervals) else np.nan,
                }
            )

    tidy = pd.DataFrame(tidy_rows)
    censoring = pd.DataFrame(censor_rows)
    summaries = pd.DataFrame(summary_rows)
    return transition_records, tidy, censoring, summaries


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else str(pd.Timestamp(value).date())
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def records_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json_ready(frame.replace({pd.NA: None}).to_dict(orient="records"))


def main() -> None:
    args = parse_args()
    stage = args.stage.expanduser().resolve()
    output = args.output.expanduser().resolve()
    wave_dir = output / "wave"
    transition_dir = output / "transitions"
    audit_dir = output / "audit"
    for directory in (output, wave_dir, transition_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    wave_frames: list[pd.DataFrame] = []
    source_audits: list[SourceAudit] = []
    invalid_id_details: list[pd.DataFrame] = []
    pain_audit_rows: list[dict[str, Any]] = []
    function_audit_rows: list[dict[str, Any]] = []
    for wave in WAVES:
        frame, audits, invalids, pain_audit, function_audit = build_wave(stage, wave)
        wave_frames.append(frame)
        source_audits.extend(audits)
        invalid_id_details.extend(invalids)
        pain_audit_rows.extend(pain_audit)
        function_audit_rows.extend(function_audit)

    long = pd.concat(wave_frames, ignore_index=True, sort=False)
    long = long.sort_values(["person_id", "wave"], kind="stable").reset_index(drop=True)
    first_death_wave = (
        long.loc[long["death_confirmed"].eq(1)].groupby("person_id")["wave"].min()
    )
    long["first_death_wave"] = long["person_id"].map(first_death_wave).astype("Int64")
    long["post_death_record"] = (
        long["first_death_wave"].notna() & long["wave"].gt(long["first_death_wave"])
    ).astype("Int64")
    long["post_death_health_record"] = (
        long["post_death_record"].eq(1) & long["health_record_present"].eq(1)
    ).astype("Int64")
    post_death = long["post_death_record"].eq(1)
    for state_col in (
        "pain_state",
        "pain_state_with_death",
        "function_state",
        "function_state_complete11",
        "joint_state",
    ):
        long.loc[post_death, state_col] = pd.NA
    long.loc[post_death, ["pain_state_num", "function_state_num"]] = pd.NA

    earliest_interview = long.groupby("person_id")["interview_date"].transform("min")
    long["time_from_first_observed_years"] = (
        pd.to_datetime(long["interview_date"]) - pd.to_datetime(earliest_interview)
    ).dt.days / 365.25

    duplicate_person_wave = int(long.duplicated(["person_id", "wave"], keep=False).sum())
    if duplicate_person_wave:
        raise ValueError("Final long data contain duplicate person-wave rows")

    transition_records, transition_tidy, transition_censoring, transition_summary = (
        build_transition_outputs(long)
    )

    state_count_rows: list[dict[str, Any]] = []
    for wave in WAVES:
        wave_data = long.loc[long["wave"].eq(wave)]
        for domain, column, states in (
            ("pain", "pain_state_with_death", PAIN_STATES),
            ("function", "function_state", FUNCTION_STATES),
            ("joint", "joint_state", JOINT_STATES),
        ):
            known = int(wave_data[column].notna().sum())
            for state in states:
                count = int(wave_data[column].eq(state).sum())
                state_count_rows.append(
                    {
                        "wave": wave,
                        "domain": domain,
                        "state": state,
                        "n": count,
                        "pct_of_known": count / known if known else np.nan,
                    }
                )
    state_counts = pd.DataFrame(state_count_rows)

    key_variables = [
        "interview_date",
        "individual_weight",
        "pain_present",
        "msk_pain_n",
        "pain_state_with_death",
        "badl_n",
        "iadl5_n",
        "function_state",
        "joint_state",
    ]
    missing_rows: list[dict[str, Any]] = []
    for wave in WAVES:
        wave_data = long.loc[long["wave"].eq(wave)]
        for variable in key_variables:
            missing = int(wave_data[variable].isna().sum())
            missing_rows.append(
                {
                    "wave": wave,
                    "variable": variable,
                    "missing_n": missing,
                    "missing_pct": missing / len(wave_data) if len(wave_data) else np.nan,
                }
            )
    missingness = pd.DataFrame(missing_rows)

    wave_summary_rows: list[dict[str, Any]] = []
    for wave in WAVES:
        data = long.loc[long["wave"].eq(wave)]
        pain_yes = data["pain_present"].eq(1)
        wave_summary_rows.append(
            {
                "wave": wave,
                "person_wave_rows": len(data),
                "health_interviews": int(data["health_record_present"].eq(1).sum()),
                "confirmed_deaths": int(data["death_confirmed"].eq(1).sum()),
                "no_health_no_death": int(data["vital_observation"].eq("no_health_no_death").sum()),
                "pain_state_known_including_death": int(data["pain_state_with_death"].notna().sum()),
                "function_state_known_including_death": int(data["function_state"].notna().sum()),
                "joint_state_known_including_death": int(data["joint_state"].notna().sum()),
                "pain_positive_no_valid_location_block": int(
                    (pain_yes & data["msk_pain_n"].isna()).sum()
                ),
                "pain_positive_zero_msk_sites": int(
                    (pain_yes & data["msk_pain_n"].eq(0)).sum()
                ),
                "badl_structural_skips_recoded_zero": int(data["badl_structural_skip"].fillna(0).sum()),
                "weight_missing": int(data["individual_weight"].isna().sum()),
                "invalid_interview_year_codes": int(data["interview_year_invalid_code"].sum()),
                "invalid_interview_month_codes": int(data["interview_month_invalid_code"].sum()),
                "invalid_died_codes": int(data["died_invalid_code"].sum()),
                "invalid_or_nonpositive_weights": int(data["weight_invalid_or_nonpositive"].sum()),
            }
        )
    wave_summary = pd.DataFrame(wave_summary_rows)

    source_audit_frame = pd.DataFrame([audit.as_dict() for audit in source_audits])
    invalid_id_frame = pd.concat(invalid_id_details, ignore_index=True)
    pain_audit_frame = pd.DataFrame(pain_audit_rows)
    function_audit_frame = pd.DataFrame(function_audit_rows)
    standardized_binary_columns = [
        *[f"pain_site_{name}" for name in MSK_SITES.values()],
        *[f"badl_{name}" for name in BADL_NAMES],
        *[f"iadl_{name}" for name in IADL_NAMES],
    ]
    standardized_binary_invalid = sum(
        int((long[column].notna() & ~long[column].isin([0, 1])).sum())
        for column in standardized_binary_columns
    )
    pain_no_but_nonzero = int(
        (long["pain_present"].eq(0) & long["msk_pain_n"].fillna(0).ne(0)).sum()
    )
    count_range_invalid = int(
        (long["msk_pain_n"].notna() & ~long["msk_pain_n"].between(0, 12)).sum()
        + (long["badl_n"].notna() & ~long["badl_n"].between(0, 6)).sum()
        + (long["iadl5_n"].notna() & ~long["iadl5_n"].between(0, 5)).sum()
    )
    general_invalid_codes = int(
        long[
            [
                "interview_year_invalid_code",
                "interview_month_invalid_code",
                "died_invalid_code",
                "weight_invalid_or_nonpositive",
            ]
        ].sum().sum()
    )
    duplicate_checks = pd.DataFrame(
        [
            {
                "check": "raw/standard source duplicate ID rows",
                "count": int(
                    source_audit_frame["duplicate_raw_id_rows"].sum()
                    + source_audit_frame["duplicate_standard_id_rows"].sum()
                ),
                "status": "PASS"
                if int(
                    source_audit_frame["duplicate_raw_id_rows"].sum()
                    + source_audit_frame["duplicate_standard_id_rows"].sum()
                )
                == 0
                else "REVIEW",
            },
            {
                "check": "final duplicate person-wave rows",
                "count": duplicate_person_wave,
                "status": "PASS" if duplicate_person_wave == 0 else "FAIL",
            },
            {
                "check": "invalid person ID rows excluded",
                "count": len(invalid_id_frame),
                "status": "REVIEW" if len(invalid_id_frame) else "PASS",
            },
            {
                "check": "same-wave health/death conflicts",
                "count": int(long["health_death_conflict"].sum()),
                "status": "PASS" if int(long["health_death_conflict"].sum()) == 0 else "REVIEW",
            },
            {
                "check": "records after first confirmed death",
                "count": int(long["post_death_record"].sum()),
                "status": "PASS" if int(long["post_death_record"].sum()) == 0 else "REVIEW",
            },
            {
                "check": "health interviews after first confirmed death",
                "count": int(long["post_death_health_record"].sum()),
                "status": "PASS" if int(long["post_death_health_record"].sum()) == 0 else "FAIL",
            },
            {
                "check": "invalid standardized 0/1 item values",
                "count": standardized_binary_invalid,
                "status": "PASS" if standardized_binary_invalid == 0 else "FAIL",
            },
            {
                "check": "out-of-range pain/BADL/IADL counts",
                "count": count_range_invalid,
                "status": "PASS" if count_range_invalid == 0 else "FAIL",
            },
            {
                "check": "screened no-pain rows with nonzero MSK count",
                "count": pain_no_but_nonzero,
                "status": "PASS" if pain_no_but_nonzero == 0 else "FAIL",
            },
            {
                "check": "invalid time/death/weight raw codes",
                "count": general_invalid_codes,
                "status": "PASS" if general_invalid_codes == 0 else "REVIEW",
            },
        ]
    )

    summary_metrics = pd.DataFrame(
        [
            {"metric": "unique_persons", "value": int(long["person_id"].nunique())},
            {"metric": "person_wave_rows", "value": len(long)},
            {"metric": "waves", "value": len(WAVES)},
            {"metric": "confirmed_death_rows", "value": int(long["death_confirmed"].sum())},
            {"metric": "duplicate_person_wave_rows", "value": duplicate_person_wave},
            {"metric": "invalid_id_rows_excluded", "value": len(invalid_id_frame)},
            {"metric": "post_death_health_records", "value": int(long["post_death_health_record"].sum())},
        ]
    )

    preferred_columns = [
        "person_id",
        "person_id_raw",
        "wave",
        "household_id",
        "community_id",
        "health_record_present",
        "sample_record_present",
        "weight_record_present",
        "exit_record_present",
        "vital_observation",
        "died_raw",
        "death_confirmed",
        "death_state",
        "first_death_wave",
        "post_death_record",
        "post_death_health_record",
        "interview_year",
        "interview_month",
        "interview_year_invalid_code",
        "interview_month_invalid_code",
        "interview_date",
        "time_from_first_observed_years",
        "individual_weight",
        "weight_invalid_or_nonpositive",
        "died_invalid_code",
        "pain_screen_raw",
        "pain_present",
        "pain_site_section_valid",
        *[f"pain_site_{name}" for name in MSK_SITES.values()],
        "msk_pain_n",
        "pain_state",
        "pain_state_num",
        "pain_state_with_death",
        *[f"badl_{name}" for name in BADL_NAMES],
        "badl_structural_skip",
        "badl_valid_n",
        "badl_n",
        *[f"iadl_{name}" for name in IADL_NAMES],
        "iadl_phone",
        "iadl5_valid_n",
        "iadl5_n",
        "function_state",
        "function_state_num",
        "all11_complete_raw",
        "function_state_complete11",
        "joint_state",
        "health_death_conflict",
    ]
    long_output = long[preferred_columns].copy()
    long_output["interview_date"] = long_output["interview_date"].dt.strftime("%Y-%m-%d")
    long_output.to_csv(
        output / "CHARLS_统一长格式数据_2011_2020_5wave.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )
    stata_output = long_output.copy()
    stata_output["interview_date"] = pd.to_datetime(stata_output["interview_date"], errors="coerce")
    for column in stata_output.columns:
        if str(stata_output[column].dtype) == "Int64":
            stata_output[column] = stata_output[column].astype("float64")
        elif str(stata_output[column].dtype) == "string":
            stata_output[column] = stata_output[column].astype(object).where(stata_output[column].notna(), None)
    stata_output.to_stata(
        output / "CHARLS_统一长格式数据_2011_2020_5wave.dta",
        write_index=False,
        version=118,
        convert_dates={"interview_date": "td"},
        variable_labels={
            "person_id": "Harmonized 12-character person ID",
            "wave": "CHARLS wave year",
            "msk_pain_n": "Number of 12 musculoskeletal pain sites",
            "pain_state": "Pain state among living interviewees: P0/P1/P2",
            "pain_state_with_death": "Pain state including confirmed death D",
            "badl_n": "Number of 6 BADL limitations; complete items required",
            "iadl5_n": "Number of 5 cross-wave IADL limitations; complete items required",
            "function_state": "Function state F0/F1/F2 or confirmed death D",
            "joint_state": "Joint pain-function state or confirmed death D",
            "vital_observation": "Interview/death/nonresponse observation classification",
        },
    )
    for wave in WAVES:
        long_output.loc[long_output["wave"].eq(wave)].to_csv(
            wave_dir / f"CHARLS_统一数据_{wave}.csv",
            index=False,
            encoding="utf-8-sig",
            na_rep="",
        )

    transition_export = transition_records.copy()
    for column in ("interview_date_from", "interview_date_to"):
        transition_export[column] = pd.to_datetime(transition_export[column]).dt.strftime("%Y-%m-%d")
    transition_export.to_csv(
        transition_dir / "CHARLS_相邻波次转移记录.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )
    for domain, chinese_name in (("pain", "疼痛"), ("function", "功能"), ("joint", "联合")):
        transition_tidy.loc[transition_tidy["domain"].eq(domain)].to_csv(
            transition_dir / f"CHARLS_{chinese_name}状态转移频数.csv",
            index=False,
            encoding="utf-8-sig",
            na_rep="",
        )
    transition_censoring.to_csv(
        transition_dir / "CHARLS_转移删失分解.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )
    invalid_id_frame.to_csv(
        audit_dir / "CHARLS_无效ID明细.csv", index=False, encoding="utf-8-sig", na_rep=""
    )
    missingness.to_csv(
        audit_dir / "CHARLS_变量缺失统计.csv", index=False, encoding="utf-8-sig", na_rep=""
    )

    payload = {
        "generated_by": str(Path(__file__).resolve()),
        "stage_folder": str(stage),
        "rules": [
            "Pain P0=0, P1=1, P2>=2 across 12 musculoskeletal sites.",
            "Function F0=no BADL/IADL, F1=IADL only, F2>=1 BADL, D=confirmed death.",
            "2011 IDs standardized by inserting 0 before the final two digits.",
            "2011/2013/2015/2018 all-six-BADL structural skips recoded to zero only when the IADL block has a valid response.",
            "died=0 without a health interview remains no_health_no_death and is censored, not alive.",
            "No age restriction applied at this cleaning stage.",
        ],
        "summary_metrics": records_json(summary_metrics),
        "source_audit": records_json(source_audit_frame),
        "wave_summary": records_json(wave_summary),
        "duplicate_checks": records_json(duplicate_checks),
        "state_counts": records_json(state_counts),
        "missingness": records_json(missingness),
        "pain_code_audit": records_json(pain_audit_frame),
        "function_code_audit": records_json(function_audit_frame),
        "transition_summary": records_json(transition_summary),
        "transition_tidy": records_json(transition_tidy),
        "transition_censoring": records_json(transition_censoring),
    }
    with (audit_dir / "CHARLS_质控明细.json").open("w", encoding="utf-8") as stream:
        json.dump(json_ready(payload), stream, ensure_ascii=False, indent=2)

    print("CHARLS multistate cleaning complete")
    print(summary_metrics.to_string(index=False))
    print("\nWave summary")
    print(wave_summary.to_string(index=False))
    print("\nDuplicate/death checks")
    print(duplicate_checks.to_string(index=False))
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
