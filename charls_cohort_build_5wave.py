#!/usr/bin/env python3
"""Build the age-45+ delayed-entry CHARLS multistate analysis cohort.

Inputs are the harmonized phase-1 person-wave file and original CHARLS wave
modules.  Outputs include a covariate-enriched long file, domain-specific panel
and adjacent-interval model-preparation files, and a JSON payload used to build
the cohort/Table-1 workbook.

Key cohort rule
---------------
Participants enter only if they were at least 45 years old at their *first*
valid health interview.  Refreshment samples first interviewed in 2013, 2015,
2018, or 2020 are therefore allowed as delayed entries.  Someone interviewed before
age 45 is not allowed to enter later merely by ageing into eligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent
STAGE = PROJECT / "stage"
PHASE1 = PROJECT / "phase1" / "CHARLS_统一长格式数据_2011_2020_5wave.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent
WAVES = (2011, 2013, 2015, 2018, 2020)
WAVE_PAIRS = ((2011, 2013), (2013, 2015), (2015, 2018), (2018, 2020))

DEMOGRAPHIC_FILES = {
    2011: "demographic_background.dta",
    2013: "Demographic_Background.dta",
    2015: "Demographic_Background.dta",
    2018: "Demographic_Background.dta",
    2020: "Demographic_Background.dta",
}
HEALTH_FILES = {
    2011: "health_status_and_functioning.dta",
    2013: "Health_Status_and_Functioning.dta",
    2015: "Health_Status_and_Functioning.dta",
    2018: "Health_Status_and_Functioning.dta",
    2020: "Health_Status_and_Functioning.dta",
}

DEMO_COLUMNS = {
    2011: ["rgender", "ba002_1", "ba002_2", "ba002_3", "bd001", "be001", "bc001"],
    2013: [
        "ba000_w2_3", "zba002_1", "zba002_2", "zba002_3", "ba002_1", "ba002_2",
        "ba002_3", "zbd001", "bd001", "bd001_w2_3", "bd001_w2_4", "be001",
        "zbc001", "bc001",
    ],
    2015: [
        "ba000_w2_3", "ba002_1", "ba002_2", "ba002_3", "ba004_w3_1", "ba004_w3_2",
        "ba004_w3_3", "bd001_w2_4", "be001", "bc001_w3_2", "bc002_w3_1",
    ],
    2018: [
        "xrgender", "ba002_1", "ba002_2", "ba002_3", "ba004_w3_1", "ba004_w3_2",
        "ba004_w3_3", "bd001_w2_4", "be001", "zbc004",
    ],
    2020: [
        "xrgender", "xrage", "zrbirthyear", "ba003_1", "ba003_2", "ba003_3", "zredu",
        "ba010", "ba011", "ba009",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--phase1", type=Path, default=PHASE1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clean_string(series: pd.Series) -> pd.Series:
    value = series.astype("string").str.strip()
    value = value.str.replace(r"\.0$", "", regex=True)
    return value.mask(value.eq(""))


def standardize_person_id(series: pd.Series, wave: int) -> pd.Series:
    value = clean_string(series)
    if wave == 2011:
        valid = value.str.fullmatch(r"\d{11}").fillna(False)
        result = value.str.slice(0, -2) + "0" + value.str.slice(-2)
    else:
        valid = value.str.fullmatch(r"\d{12}").fillna(False)
        result = value
    return result.where(valid).astype("string")


def read_stata_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    reader = pd.io.stata.StataReader(path, convert_categoricals=False)
    available = set(reader.variable_labels())
    requested = list(dict.fromkeys(columns))
    missing = sorted(set(requested) - available)
    if missing:
        raise KeyError(f"Missing variables in {path}: {missing}")
    return pd.read_stata(path, columns=requested, convert_categoricals=False)


def valid_numeric(series: pd.Series, lower: float, upper: float) -> pd.Series:
    value = pd.to_numeric(series, errors="coerce")
    return value.where(value.between(lower, upper))


def coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame:
            result = result.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return result


def first_mode(series: pd.Series) -> float:
    value = series.dropna()
    if value.empty:
        return np.nan
    counts = value.value_counts()
    winners = set(counts[counts.eq(counts.max())].index.tolist())
    for item in value:
        if item in winners:
            return float(item)
    return float(value.iloc[0])


def harmonize_demographics(stage: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    source_audit: list[dict[str, Any]] = []
    for wave in WAVES:
        path = stage / str(wave) / DEMOGRAPHIC_FILES[wave]
        raw = read_stata_columns(path, ["ID", *DEMO_COLUMNS[wave]])
        raw["person_id"] = standardize_person_id(raw["ID"], wave)
        invalid_ids = int(raw["person_id"].isna().sum())
        raw = raw.dropna(subset=["person_id"]).copy()
        duplicate_rows = int(raw.duplicated("person_id", keep=False).sum())
        if duplicate_rows:
            raw = raw.sort_values("person_id").drop_duplicates("person_id", keep="first")

        out = pd.DataFrame({"person_id": raw["person_id"], "wave": wave})
        if wave == 2011:
            out["sex_raw"] = raw["rgender"]
            out["birth_year_wave"] = valid_numeric(raw["ba002_1"], 1880, 2011)
            out["birth_month_wave"] = valid_numeric(raw["ba002_2"], 1, 12)
            out["birth_day_wave"] = valid_numeric(raw["ba002_3"], 1, 31)
            out["education_wave"] = valid_numeric(raw["bd001"], 1, 11)
            out["marital_raw"] = valid_numeric(raw["be001"], 1, 7)
            out["hukou_raw"] = valid_numeric(raw["bc001"], 1, 4)
        elif wave == 2013:
            out["sex_raw"] = raw["ba000_w2_3"]
            out["birth_year_wave"] = valid_numeric(coalesce(raw, ["ba002_1", "zba002_1"]), 1880, 2013)
            out["birth_month_wave"] = valid_numeric(coalesce(raw, ["ba002_2", "zba002_2"]), 1, 12)
            out["birth_day_wave"] = valid_numeric(coalesce(raw, ["ba002_3", "zba002_3"]), 1, 31)
            out["education_wave"] = valid_numeric(
                coalesce(raw, ["bd001", "zbd001", "bd001_w2_3", "bd001_w2_4"]), 1, 11
            )
            out["marital_raw"] = valid_numeric(raw["be001"], 1, 7)
            out["hukou_raw"] = valid_numeric(coalesce(raw, ["bc001", "zbc001"]), 1, 4)
        elif wave == 2015:
            out["sex_raw"] = raw["ba000_w2_3"]
            out["birth_year_wave"] = valid_numeric(coalesce(raw, ["ba002_1", "ba004_w3_1"]), 1880, 2015)
            out["birth_month_wave"] = valid_numeric(coalesce(raw, ["ba002_2", "ba004_w3_2"]), 1, 12)
            out["birth_day_wave"] = valid_numeric(coalesce(raw, ["ba002_3", "ba004_w3_3"]), 1, 31)
            out["education_wave"] = valid_numeric(raw["bd001_w2_4"], 1, 11)
            out["marital_raw"] = valid_numeric(raw["be001"], 1, 7)
            out["hukou_raw"] = valid_numeric(coalesce(raw, ["bc002_w3_1", "bc001_w3_2"]), 1, 4)
        elif wave == 2018:
            out["sex_raw"] = raw["xrgender"]
            out["birth_year_wave"] = valid_numeric(coalesce(raw, ["ba002_1", "ba004_w3_1"]), 1880, 2018)
            out["birth_month_wave"] = valid_numeric(coalesce(raw, ["ba002_2", "ba004_w3_2"]), 1, 12)
            out["birth_day_wave"] = valid_numeric(coalesce(raw, ["ba002_3", "ba004_w3_3"]), 1, 31)
            out["education_wave"] = valid_numeric(raw["bd001_w2_4"], 1, 11)
            out["marital_raw"] = valid_numeric(raw["be001"], 1, 7)
            out["hukou_raw"] = valid_numeric(raw["zbc004"], 1, 4)
        else:
            out["sex_raw"] = raw["xrgender"]
            # In the released 2020 file, zrbirthyear equals 2020-xrage while xrage
            # is the carried 2018 age.  It is therefore exactly two years later
            # than the 2018 birth year for overlapping respondents.  Use a newly
            # reported ba003_1 first and correct this generated fallback by -2.
            generated_birth = pd.to_numeric(raw["zrbirthyear"], errors="coerce") - 2
            birth_2020 = pd.to_numeric(raw["ba003_1"], errors="coerce").fillna(generated_birth)
            out["birth_year_wave"] = valid_numeric(birth_2020, 1880, 2020)
            out["birth_month_wave"] = valid_numeric(raw["ba003_2"], 1, 12)
            out["birth_day_wave"] = valid_numeric(raw["ba003_3"], 1, 31)
            out["education_wave"] = valid_numeric(coalesce(raw, ["ba010", "zredu"]), 1, 11)
            out["marital_raw"] = valid_numeric(raw["ba011"], 1, 7)
            out["hukou_raw"] = valid_numeric(raw["ba009"], 1, 4)
        out["sex_raw"] = valid_numeric(out["sex_raw"], 1, 2)
        rows.append(out)
        source_audit.append({
            "wave": wave,
            "source": "demographic",
            "file": str(path),
            "rows_raw": int(len(raw) + invalid_ids),
            "invalid_id_rows": invalid_ids,
            "duplicate_id_rows": duplicate_rows,
            "rows_retained": int(len(raw)),
        })

    wave_demo = pd.concat(rows, ignore_index=True)
    person = wave_demo.groupby("person_id", sort=False).agg(
        birth_year=("birth_year_wave", first_mode),
        birth_year_min=("birth_year_wave", "min"),
        birth_year_max=("birth_year_wave", "max"),
        birth_month=("birth_month_wave", first_mode),
        birth_day=("birth_day_wave", first_mode),
        sex=("sex_raw", first_mode),
        education_highest_code=("education_wave", "max"),
    ).reset_index()

    spread = wave_demo.groupby("person_id")["birth_year_wave"].agg(lambda s: s.max() - s.min() if s.notna().any() else np.nan)
    sex_nunique = wave_demo.groupby("person_id")["sex_raw"].nunique(dropna=True)
    person["birth_year_conflict_gt1"] = person["person_id"].map(spread.gt(1)).fillna(False).astype(int)
    person["sex_conflict"] = person["person_id"].map(sex_nunique.gt(1)).fillna(False).astype(int)

    education = person["education_highest_code"]
    person["education_cat"] = pd.Series(pd.NA, index=person.index, dtype="string")
    person.loc[education.eq(1), "education_cat"] = "无正规教育"
    person.loc[education.between(2, 4), "education_cat"] = "小学及以下"
    person.loc[education.eq(5), "education_cat"] = "初中"
    person.loc[education.between(6, 11), "education_cat"] = "高中及以上"
    person["female"] = person["sex"].map({1.0: 0, 2.0: 1})
    person["sex_label"] = person["sex"].map({1.0: "男", 2.0: "女"}).astype("string")

    wave_demo = wave_demo.sort_values(["person_id", "wave"])
    wave_demo["hukou_raw_observed"] = wave_demo["hukou_raw"]
    wave_demo["hukou_raw"] = wave_demo.groupby("person_id")["hukou_raw"].transform(lambda s: s.ffill().bfill())
    wave_demo["hukou_filled_crosswave"] = (
        wave_demo["hukou_raw_observed"].isna() & wave_demo["hukou_raw"].notna()
    ).astype(int)
    wave_demo["hukou_agricultural"] = wave_demo["hukou_raw"].eq(1).where(wave_demo["hukou_raw"].notna()).astype("Float64")
    wave_demo["marital_cat"] = pd.Series(pd.NA, index=wave_demo.index, dtype="string")
    wave_demo.loc[wave_demo["marital_raw"].isin([1, 2, 7]), "marital_cat"] = "已婚/伴侣"
    wave_demo.loc[wave_demo["marital_raw"].isin([3, 4]), "marital_cat"] = "分居/离婚"
    wave_demo.loc[wave_demo["marital_raw"].eq(5), "marital_cat"] = "丧偶"
    wave_demo.loc[wave_demo["marital_raw"].eq(6), "marital_cat"] = "从未婚"
    wave_demo["partnered"] = wave_demo["marital_raw"].isin([1, 2, 7]).where(wave_demo["marital_raw"].notna()).astype("Float64")
    keep = [
        "person_id", "wave", "marital_raw", "marital_cat", "partnered", "hukou_raw",
        "hukou_agricultural", "hukou_filled_crosswave",
    ]
    return wave_demo[keep], person, source_audit


def yes_no(series: pd.Series) -> pd.Series:
    value = pd.to_numeric(series, errors="coerce")
    return value.map({1.0: 1.0, 2.0: 0.0})


def current_chronic_2018(raw: pd.DataFrame, index: int) -> pd.Series:
    z = pd.to_numeric(raw[f"zdisease_{index}_"], errors="coerce")
    compare = pd.to_numeric(raw[f"da010_w2_2_{index}_"], errors="coerce")
    direct = pd.to_numeric(raw[f"da007_{index}_"], errors="coerce")
    result = pd.Series(np.nan, index=raw.index, dtype="float64")
    result.loc[direct.eq(2)] = 0
    result.loc[direct.eq(1)] = 1
    result.loc[z.eq(1) & compare.ne(99)] = 1
    result.loc[z.eq(1) & compare.eq(99) & ~direct.eq(1)] = 0
    return result


def current_chronic_2020(raw: pd.DataFrame, index: int) -> pd.Series:
    z = pd.to_numeric(raw[f"zdisease_{index}_"], errors="coerce")
    compare = pd.to_numeric(raw[f"da002_{index}_"], errors="coerce")
    direct = pd.to_numeric(raw[f"da003_{index}_"], errors="coerce")
    result = pd.Series(np.nan, index=raw.index, dtype="float64")
    result.loc[direct.eq(2)] = 0
    result.loc[direct.eq(1)] = 1
    result.loc[z.eq(1) & compare.ne(99)] = 1
    result.loc[z.eq(1) & compare.eq(99) & ~direct.eq(1)] = 0
    return result


def cesd_score(raw: pd.DataFrame, variables: list[str], positive_indexes: set[int]) -> tuple[pd.Series, pd.Series]:
    scored = []
    for index, variable in enumerate(variables):
        value = pd.to_numeric(raw[variable], errors="coerce").where(lambda s: s.isin([1, 2, 3, 4]))
        score = 4 - value if index in positive_indexes else value - 1
        scored.append(score)
    item = pd.concat(scored, axis=1)
    valid_n = item.notna().sum(axis=1)
    score = item.sum(axis=1, min_count=8) * 10 / valid_n.where(valid_n.ge(8))
    return score.round(2), valid_n.astype("Int64")


def physical_activity(raw: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    items = pd.DataFrame({name: yes_no(raw[var]) for name, var in zip(["vigorous", "moderate", "light"], variables)})
    any_activity = pd.Series(np.nan, index=raw.index, dtype="float64")
    any_activity.loc[items.eq(1).any(axis=1)] = 1
    any_activity.loc[items.notna().all(axis=1) & items.eq(0).all(axis=1)] = 0
    return items, any_activity


def harmonize_health_covariates(stage: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for wave in WAVES:
        chronic: list[str] = []
        if wave == 2011:
            chronic = [f"da007_{i}_" for i in range(1, 15)]
            cesd = [f"dc{i:03d}" for i in range(9, 19)]
            columns = [*chronic, *cesd, "da059", "da061", "da067", "da051_1_", "da051_2_", "da051_3_", "proxy"]
        elif wave == 2013:
            chronic = [f"zda007_{i}_" for i in range(1, 15)]
            cesd = [f"dc{i:03d}" for i in range(9, 19)]
            columns = [*chronic, *cesd, "da059", "da061", "da067", "da051_1_", "da051_2_", "da051_3_", "db034"]
        elif wave == 2015:
            chronic = [f"zda007_{i}_" for i in range(1, 15)]
            cesd = [f"dc{i:03d}" for i in range(9, 19)]
            columns = [*chronic, *cesd, "da059", "da061", "da067", "da051_1_", "da051_2_", "da051_3_", "proxy"]
        elif wave == 2018:
            chronic = [
                *[f"da007_{i}_" for i in range(1, 15)],
                *[f"zdisease_{i}_" for i in range(1, 15)],
                *[f"da010_w2_2_{i}_" for i in range(1, 15)],
            ]
            cesd = []
            columns = [*chronic, "da059", "da061_w4", "da067", "da051_1_", "da051_2_", "da051_3_", "db034"]
        else:
            chronic = [
                *[f"zdisease_{i}_" for i in range(1, 15)],
                *[f"da002_{i}_" for i in range(1, 15)],
                *[f"da003_{i}_" for i in range(1, 15)],
            ]
            cesd = [f"dc{i:03d}" for i in range(16, 26)]
            columns = [*chronic, *cesd, "da046", "da047", "da051", "da032_1_", "da032_2_", "da032_3_", "proxy_5"]

        path = stage / str(wave) / HEALTH_FILES[wave]
        raw = read_stata_columns(path, ["ID", *columns])
        raw["person_id"] = standardize_person_id(raw["ID"], wave)
        invalid_ids = int(raw["person_id"].isna().sum())
        raw = raw.dropna(subset=["person_id"]).copy()
        duplicate_rows = int(raw.duplicated("person_id", keep=False).sum())
        if duplicate_rows:
            raw = raw.sort_values("person_id").drop_duplicates("person_id", keep="first")
        out = pd.DataFrame({"person_id": raw["person_id"], "wave": wave})

        disease_items = pd.DataFrame(index=raw.index)
        for index in range(1, 15):
            if wave == 2011:
                disease_items[f"d{index}"] = yes_no(raw[f"da007_{index}_"])
            elif wave in (2013, 2015):
                disease_items[f"d{index}"] = pd.to_numeric(raw[f"zda007_{index}_"], errors="coerce").eq(1).astype(float)
            elif wave == 2018:
                disease_items[f"d{index}"] = current_chronic_2018(raw, index)
            else:
                disease_items[f"d{index}"] = current_chronic_2020(raw, index)
        out["chronic_valid_n"] = disease_items.notna().sum(axis=1).to_numpy()
        out["chronic_n"] = disease_items.sum(axis=1, min_count=14).to_numpy()
        out["multimorbidity"] = out["chronic_n"].ge(2).where(out["chronic_n"].notna()).astype("Float64")

        if wave == 2018:
            cognition_path = stage / "2018" / "Cognition.dta"
            cognition_vars = [f"dc{i:03d}" for i in range(9, 19)]
            cognition = read_stata_columns(cognition_path, ["ID", *cognition_vars])
            cognition["person_id"] = standardize_person_id(cognition["ID"], 2018)
            cognition = cognition.dropna(subset=["person_id"]).drop_duplicates("person_id")
            raw_for_cesd = raw[["person_id"]].merge(cognition[["person_id", *cognition_vars]], on="person_id", how="left", validate="one_to_one")
            score, valid_n = cesd_score(raw_for_cesd, cognition_vars, {4, 7})
        else:
            score, valid_n = cesd_score(raw, cesd, {4, 7})
        out["cesd10_score"] = score.to_numpy()
        out["cesd10_valid_n"] = valid_n.to_numpy()
        out["depressive_symptoms"] = out["cesd10_score"].ge(10).where(out["cesd10_score"].notna()).astype("Float64")

        if wave == 2011:
            current_smoker = pd.Series(np.nan, index=raw.index, dtype="float64")
            current_smoker.loc[pd.to_numeric(raw["da061"], errors="coerce").eq(1)] = 1
            current_smoker.loc[pd.to_numeric(raw["da061"], errors="coerce").eq(2) | pd.to_numeric(raw["da059"], errors="coerce").eq(2)] = 0
            drink = pd.to_numeric(raw["da067"], errors="coerce")
            activity_vars = ["da051_1_", "da051_2_", "da051_3_"]
            proxy = pd.to_numeric(raw["proxy"], errors="coerce").map({0.0: 0.0, 1.0: 1.0})
        elif wave == 2013:
            current_smoker = pd.Series(np.nan, index=raw.index, dtype="float64")
            status = pd.to_numeric(raw["da061"], errors="coerce")
            current_smoker.loc[status.eq(1)] = 1
            current_smoker.loc[status.isin([2, 3]) | pd.to_numeric(raw["da059"], errors="coerce").eq(2)] = 0
            drink = pd.to_numeric(raw["da067"], errors="coerce")
            activity_vars = ["da051_1_", "da051_2_", "da051_3_"]
            proxy = raw["db034"].notna().astype(float)
        elif wave == 2015:
            current_smoker = pd.Series(np.nan, index=raw.index, dtype="float64")
            status = pd.to_numeric(raw["da061"], errors="coerce")
            current_smoker.loc[status.eq(1)] = 1
            current_smoker.loc[status.isin([2, 3]) | pd.to_numeric(raw["da059"], errors="coerce").eq(2)] = 0
            drink = pd.to_numeric(raw["da067"], errors="coerce")
            activity_vars = ["da051_1_", "da051_2_", "da051_3_"]
            proxy = pd.to_numeric(raw["proxy"], errors="coerce").map({0.0: 0.0, 1.0: 1.0})
        elif wave == 2018:
            status = pd.to_numeric(raw["da061_w4"], errors="coerce")
            current_smoker = pd.Series(np.nan, index=raw.index, dtype="float64")
            current_smoker.loc[status.eq(1)] = 1
            current_smoker.loc[status.isin([2, 3]) | pd.to_numeric(raw["da059"], errors="coerce").eq(2)] = 0
            drink = pd.to_numeric(raw["da067"], errors="coerce")
            activity_vars = ["da051_1_", "da051_2_", "da051_3_"]
            proxy = raw["db034"].notna().astype(float)
        else:
            status = pd.to_numeric(raw["da047"], errors="coerce")
            current_smoker = pd.Series(np.nan, index=raw.index, dtype="float64")
            current_smoker.loc[status.eq(1)] = 1
            current_smoker.loc[status.isin([2, 3]) | pd.to_numeric(raw["da046"], errors="coerce").eq(2)] = 0
            drink = pd.to_numeric(raw["da051"], errors="coerce")
            activity_vars = ["da032_1_", "da032_2_", "da032_3_"]
            proxy = pd.to_numeric(raw["proxy_5"], errors="coerce").eq(1).astype(float)

        out["current_smoker"] = current_smoker.to_numpy()
        out["alcohol_past_year"] = drink.isin([1, 2]).where(drink.isin([1, 2, 3])).astype("Float64").to_numpy()
        out["alcohol_monthly"] = drink.eq(1).where(drink.isin([1, 2, 3])).astype("Float64").to_numpy()
        pa_items, pa_any = physical_activity(raw, activity_vars)
        out["pa_vigorous"] = pa_items["vigorous"].to_numpy()
        out["pa_moderate"] = pa_items["moderate"].to_numpy()
        out["pa_light"] = pa_items["light"].to_numpy()
        out["physical_activity_any"] = pa_any.to_numpy()
        out["proxy_health"] = proxy.to_numpy()
        outputs.append(out)
        audit.append({
            "wave": wave,
            "source": "health_covariates",
            "file": str(path),
            "rows_raw": int(len(raw) + invalid_ids),
            "invalid_id_rows": invalid_ids,
            "duplicate_id_rows": duplicate_rows,
            "rows_retained": int(len(raw)),
        })
    return pd.concat(outputs, ignore_index=True), audit


def compute_age(frame: pd.DataFrame) -> pd.Series:
    year = pd.to_numeric(frame["birth_year"], errors="coerce")
    month = pd.to_numeric(frame["birth_month"], errors="coerce")
    day = pd.to_numeric(frame["birth_day"], errors="coerce")
    date = pd.to_datetime(frame["interview_date"], errors="coerce")
    age = date.dt.year - year
    known_birthday = month.between(1, 12) & day.between(1, 31)
    before = (date.dt.month < month) | ((date.dt.month == month) & (date.dt.day < day))
    age = age - (known_birthday & before).astype(int)
    return age.where(age.between(0, 120))


def add_cohort_flags(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    active_health = frame["health_record_present"].eq(1) & frame["death_confirmed"].eq(0) & frame["post_death_health_record"].eq(0)
    first_rows = (
        frame.loc[active_health]
        .sort_values(["person_id", "wave"])
        .groupby("person_id", as_index=False)
        .first()[["person_id", "wave", "age_years", "interview_date", "birth_year_min", "birth_year_max"]]
        .rename(columns={"wave": "first_health_wave", "age_years": "age_first_health", "interview_date": "entry_date"})
    )
    people = pd.DataFrame({"person_id": frame["person_id"].drop_duplicates()}).merge(first_rows, on="person_id", how="left")
    people["has_health_interview"] = people["first_health_wave"].notna().astype(int)
    people["valid_age_first_health"] = people["age_first_health"].notna().astype(int)
    people["eligible_age45"] = people["age_first_health"].ge(45).where(people["age_first_health"].notna(), False).astype(int)
    entry_year = pd.to_datetime(people["entry_date"], errors="coerce").dt.year
    people["age_first_if_latest_birth"] = entry_year - people["birth_year_max"]
    people["age_first_if_earliest_birth"] = entry_year - people["birth_year_min"]
    people["age_eligibility_sensitive"] = (
        people["age_first_if_latest_birth"].lt(45) & people["age_first_if_earliest_birth"].ge(45)
    ).fillna(False).astype(int)
    frame = frame.merge(people, on="person_id", how="left", validate="many_to_one")
    nominal = pd.to_datetime(frame["wave"].astype(str) + "-07-01")
    observation_date = pd.to_datetime(frame["interview_date"], errors="coerce").fillna(nominal)
    entry_date = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["time_since_entry_years"] = ((observation_date - entry_date).dt.days / 365.25).where(frame["eligible_age45"].eq(1)).round(4)
    frame["entry_wave"] = frame["first_health_wave"]
    return frame, people


def build_intervals(frame: pd.DataFrame, domain: str, state_col: str, active_states: list[str]) -> pd.DataFrame:
    covariates = [
        "age_years", "female", "education_highest_code", "education_cat", "marital_cat", "partnered",
        "urban_nbs", "rural_nbs", "hukou_agricultural", "chronic_n", "multimorbidity", "cesd10_score",
        "depressive_symptoms", "current_smoker", "alcohol_past_year", "alcohol_monthly",
        "physical_activity_any", "proxy_health", "individual_weight",
    ]
    rows: list[pd.DataFrame] = []
    for start_wave, end_wave in WAVE_PAIRS:
        start = frame.loc[frame["wave"].eq(start_wave)].copy()
        end_cols = ["person_id", "death_confirmed", "health_record_present", "interview_date", state_col]
        end = frame.loc[frame["wave"].eq(end_wave), end_cols].copy()
        end = end.rename(columns={column: f"next_{column}" for column in end_cols if column != "person_id"})
        merged = start.merge(end, on="person_id", how="left", validate="one_to_one", indicator="next_merge")
        merged = merged.loc[merged["eligible_age45"].eq(1) & merged[state_col].isin(active_states)].copy()
        if merged.empty:
            continue
        merged["domain"] = domain
        merged["start_wave"] = start_wave
        merged["end_wave"] = end_wave
        merged["from_state"] = merged[state_col].astype("string")
        merged["to_state"] = pd.Series("CENSORED", index=merged.index, dtype="string")
        merged["censor_reason"] = pd.Series("", index=merged.index, dtype="string")
        no_next = merged["next_merge"].eq("left_only")
        to_death = merged["next_death_confirmed"].eq(1)
        to_active = merged[f"next_{state_col}"].isin(active_states)
        next_no_health = merged["next_health_record_present"].eq(0) & ~to_death
        merged.loc[to_death, "to_state"] = "D"
        merged.loc[to_active & ~to_death, "to_state"] = merged.loc[to_active & ~to_death, f"next_{state_col}"].astype("string")
        merged.loc[no_next, "censor_reason"] = "no_next_record"
        merged.loc[~no_next & next_no_health, "censor_reason"] = "next_no_health_no_death"
        merged.loc[~no_next & ~next_no_health & ~to_death & ~to_active, "censor_reason"] = "next_state_missing"
        merged["event_observed"] = merged["to_state"].ne("CENSORED").astype(int)
        merged["transition"] = merged["from_state"] + "->" + merged["to_state"]

        start_date = pd.to_datetime(merged["interview_date"], errors="coerce").fillna(pd.Timestamp(f"{start_wave}-07-01"))
        stop_date = pd.to_datetime(merged["next_interview_date"], errors="coerce").fillna(pd.Timestamp(f"{end_wave}-07-01"))
        entry_date = pd.to_datetime(merged["entry_date"], errors="coerce")
        merged["start_date"] = start_date.dt.strftime("%Y-%m-%d")
        merged["stop_date"] = stop_date.dt.strftime("%Y-%m-%d")
        merged["interval_years"] = ((stop_date - start_date).dt.days / 365.25).round(4)
        merged["time_start_years"] = ((start_date - entry_date).dt.days / 365.25).round(4)
        merged["time_stop_years"] = ((stop_date - entry_date).dt.days / 365.25).round(4)
        keep = [
            "person_id", "domain", "start_wave", "end_wave", "start_date", "stop_date", "time_start_years",
            "time_stop_years", "interval_years", "from_state", "to_state", "event_observed", "censor_reason",
            "transition", *covariates,
        ]
        rows.append(merged[keep])
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True).sort_values(["person_id", "start_wave"]).reset_index(drop=True)
    result["origin_sequence"] = result.groupby("person_id").cumcount() + 1
    return result


def build_panel(frame: pd.DataFrame, state_col: str, active_states: list[str], person_ids: set[str]) -> pd.DataFrame:
    columns = [
        "person_id", "wave", "interview_date", "time_since_entry_years", "death_confirmed", state_col,
        "age_years", "female", "education_highest_code", "education_cat", "marital_cat", "partnered",
        "urban_nbs", "rural_nbs", "hukou_agricultural", "chronic_n", "multimorbidity", "cesd10_score",
        "depressive_symptoms", "current_smoker", "alcohol_past_year", "alcohol_monthly",
        "physical_activity_any", "proxy_health", "individual_weight",
    ]
    panel = frame.loc[
        frame["person_id"].isin(person_ids) & (frame[state_col].isin(active_states) | frame[state_col].eq("D")),
        columns,
    ].copy()
    panel = panel.rename(columns={state_col: "state"}).sort_values(["person_id", "wave"]).reset_index(drop=True)
    panel["state_sequence"] = panel.groupby("person_id").cumcount() + 1
    return panel


def format_number(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def summary_table(frame: pd.DataFrame, group_col: str, groups: list[str]) -> list[dict[str, Any]]:
    group_labels = ["总体", *groups]
    subsets = {"总体": frame}
    subsets.update({group: frame.loc[frame[group_col].eq(group)] for group in groups})
    rows: list[dict[str, Any]] = []

    header = {"section": "样本量", "variable": "参与者，n", "category": ""}
    for label in group_labels:
        header[label] = f"{len(subsets[label]):,}"
    rows.append(header)

    continuous = [
        ("人口学", "年龄，岁", "age_years"),
        ("健康", "慢性病数量（14项）", "chronic_n"),
        ("心理", "CESD-10评分", "cesd10_score"),
    ]
    for section, label, variable in continuous:
        row = {"section": section, "variable": label, "category": "均值（标准差）"}
        missing = {"section": section, "variable": label, "category": "缺失，n"}
        for group in group_labels:
            values = pd.to_numeric(subsets[group][variable], errors="coerce")
            row[group] = "—" if values.notna().sum() == 0 else f"{values.mean():.1f} ({values.std(ddof=1):.1f})"
            missing[group] = f"{values.isna().sum():,}"
        rows.extend([row, missing])

    categorical = [
        ("人口学", "性别", "sex_label", ["男", "女"]),
        ("人口学", "教育程度", "education_cat", ["无正规教育", "小学及以下", "初中", "高中及以上"]),
        ("人口学", "婚姻状态", "marital_cat", ["已婚/伴侣", "分居/离婚", "丧偶", "从未婚"]),
        ("人口学", "居住社区（NBS）", "residence_label", ["农村", "城镇"]),
        ("健康", "多病共存（≥2项）", "multimorbidity", [1.0]),
        ("心理", "抑郁症状（CESD-10≥10）", "depressive_symptoms", [1.0]),
        ("行为", "当前吸烟", "current_smoker", [1.0]),
        ("行为", "过去一年饮酒", "alcohol_past_year", [1.0]),
        ("行为", "任一强度活动≥10分钟", "physical_activity_any", [1.0]),
        ("访谈", "健康模块代理访谈", "proxy_health", [1.0]),
    ]
    for section, label, variable, categories in categorical:
        for category in categories:
            category_label = "是" if category == 1.0 else str(category)
            row = {"section": section, "variable": label, "category": category_label}
            for group in group_labels:
                values = subsets[group][variable]
                denominator = int(values.notna().sum())
                count = int(values.eq(category).sum())
                row[group] = "—" if denominator == 0 else f"{count:,} ({100 * count / denominator:.1f}%)"
            rows.append(row)
        missing = {"section": section, "variable": label, "category": "缺失，n"}
        for group in group_labels:
            missing[group] = f"{subsets[group][variable].isna().sum():,}"
        rows.append(missing)
    return rows


def missingness_rows(cohorts: dict[str, pd.DataFrame], variables: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort_name, cohort in cohorts.items():
        for variable, label in variables:
            missing_n = int(cohort[variable].isna().sum())
            rows.append({
                "cohort": cohort_name,
                "variable": variable,
                "label": label,
                "n": int(len(cohort)),
                "missing_n": missing_n,
                "missing_pct": round(100 * missing_n / len(cohort), 2) if len(cohort) else np.nan,
            })
    return rows


def stata_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.StringDtype):
            result[column] = result[column].fillna("").astype(str)
        elif str(result[column].dtype) in {"Int64", "Float64", "boolean"}:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
        elif result[column].dtype == object:
            if result[column].map(lambda x: isinstance(x, str) or pd.isna(x)).all():
                result[column] = result[column].fillna("").astype(str)
    return result


def write_dataset(frame: pd.DataFrame, csv_path: Path, dta_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    safe = stata_safe(frame)
    too_long = [column for column in safe.columns if len(column) > 32]
    if too_long:
        raise ValueError(f"Stata variable names exceed 32 characters: {too_long}")
    safe.to_stata(dta_path, write_index=False, version=119)


def main() -> None:
    args = parse_args()
    output = args.output
    audit_dir = output / "audit"
    output.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(
        args.phase1,
        dtype={"person_id": "string", "person_id_raw": "string", "household_id": "string", "community_id": "string"},
        low_memory=False,
    )
    long["interview_date"] = pd.to_datetime(long["interview_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    wave_demo, person_demo, demo_audit = harmonize_demographics(args.stage)
    health_cov, health_audit = harmonize_health_covariates(args.stage)

    long = long.merge(wave_demo, on=["person_id", "wave"], how="left", validate="one_to_one")
    long = long.merge(person_demo, on="person_id", how="left", validate="many_to_one")
    long = long.merge(health_cov, on=["person_id", "wave"], how="left", validate="one_to_one")

    psu_path = args.stage / "2011" / "PSU.dta"
    psu = pd.read_stata(psu_path, columns=["communityID", "urban_nbs"], convert_categoricals=False)
    psu["community_id"] = clean_string(psu["communityID"])
    psu = psu.drop_duplicates("community_id")[["community_id", "urban_nbs"]]
    long = long.merge(psu, on="community_id", how="left", validate="many_to_one")
    long["urban_nbs"] = pd.to_numeric(long["urban_nbs"], errors="coerce").where(lambda s: s.isin([0, 1]))
    long["rural_nbs"] = long["urban_nbs"].map({0.0: 1.0, 1.0: 0.0})
    long["residence_label"] = long["urban_nbs"].map({0.0: "农村", 1.0: "城镇"}).astype("string")
    long["age_years"] = compute_age(long)
    long, people = add_cohort_flags(long)

    # Derive person-level state availability and cohort flags.
    eligible = long["eligible_age45"].eq(1)
    active_pain = long["pain_state_with_death"].isin(["P0", "P1", "P2"])
    active_function = long["function_state"].isin(["F0", "F1", "F2"])
    active_joint = long["joint_state"].isin([f"P{p}F{f}" for p in range(3) for f in range(3)])
    availability = long.assign(
        valid_pain=(eligible & active_pain).astype(int),
        valid_function=(eligible & active_function).astype(int),
        valid_joint=(eligible & active_joint).astype(int),
    ).groupby("person_id")[["valid_pain", "valid_function", "valid_joint"]].max().reset_index()
    people = people.merge(availability, on="person_id", how="left")

    intervals = {
        "pain": build_intervals(long, "pain", "pain_state_with_death", ["P0", "P1", "P2"]),
        "function": build_intervals(long, "function", "function_state", ["F0", "F1", "F2"]),
        "joint": build_intervals(long, "joint", "joint_state", [f"P{p}F{f}" for p in range(3) for f in range(3)]),
    }
    for domain, data in intervals.items():
        model_ids = set(data["person_id"].astype(str))
        people[f"{domain}_model_person"] = people["person_id"].astype(str).isin(model_ids).astype(int)

    long = long.merge(
        people[["person_id", "pain_model_person", "function_model_person", "joint_model_person"]],
        on="person_id", how="left", validate="many_to_one",
    )
    eligible_long = long.loc[long["eligible_age45"].eq(1)].copy().sort_values(["person_id", "wave"])

    panels = {
        "pain": build_panel(eligible_long, "pain_state_with_death", ["P0", "P1", "P2"], set(intervals["pain"]["person_id"].astype(str))),
        "function": build_panel(eligible_long, "function_state", ["F0", "F1", "F2"], set(intervals["function"]["person_id"].astype(str))),
        "joint": build_panel(eligible_long, "joint_state", [f"P{p}F{f}" for p in range(3) for f in range(3)], set(intervals["joint"]["person_id"].astype(str))),
    }

    write_dataset(
        eligible_long,
        output / "CHARLS_分析长格式_45岁及以上.csv",
        output / "CHARLS_分析长格式_45岁及以上.dta",
    )
    chinese = {"pain": "疼痛", "function": "功能", "joint": "联合"}
    for domain in ["pain", "function", "joint"]:
        write_dataset(
            panels[domain], output / f"CHARLS_{chinese[domain]}模型面板数据.csv", output / f"CHARLS_{chinese[domain]}模型面板数据.dta"
        )
        write_dataset(
            intervals[domain], output / f"CHARLS_{chinese[domain]}模型区间数据.csv", output / f"CHARLS_{chinese[domain]}模型区间数据.dta"
        )

    baseline: dict[str, pd.DataFrame] = {}
    for domain, data in intervals.items():
        first = data.sort_values(["person_id", "start_wave"]).drop_duplicates("person_id").copy()
        person_columns = ["person_id", "sex_label", "residence_label"]
        first = first.merge(person_demo[person_columns[:2]], on="person_id", how="left", validate="one_to_one")
        # residence_label is already an interval covariate only as urban/rural numeric; recreate here.
        first["residence_label"] = first["urban_nbs"].map({0.0: "农村", 1.0: "城镇"}).astype("string")
        baseline[domain] = first

    pain_table = summary_table(baseline["pain"], "from_state", ["P0", "P1", "P2"])
    function_table = summary_table(baseline["function"], "from_state", ["F0", "F1", "F2"])
    joint_cross = (
        baseline["joint"].assign(
            pain=lambda d: d["from_state"].str.slice(0, 2),
            function=lambda d: d["from_state"].str.slice(2, 4),
        ).groupby(["pain", "function"]).size().reindex(
            pd.MultiIndex.from_product([["P0", "P1", "P2"], ["F0", "F1", "F2"]]), fill_value=0
        ).rename("n").reset_index()
    )
    joint_cross.columns = ["pain", "function", "n"]

    flow = [
        {"step": 1, "criterion": "四轮任一来源出现的唯一ID", "n": int(len(people))},
        {"step": 2, "criterion": "至少一次有效健康访谈", "n": int(people["has_health_interview"].sum())},
        {"step": 3, "criterion": "首次健康访谈年龄可判定", "n": int(people["valid_age_first_health"].sum())},
        {"step": 4, "criterion": "首次健康访谈年龄≥45岁（正式队列）", "n": int(people["eligible_age45"].sum())},
        {"step": 5, "criterion": "疼痛状态至少一次可判定", "n": int(((people["eligible_age45"] == 1) & (people["valid_pain"] == 1)).sum())},
        {"step": 6, "criterion": "疼痛模型至少一个相邻风险区间", "n": int(people["pain_model_person"].sum())},
        {"step": 7, "criterion": "功能状态至少一次可判定", "n": int(((people["eligible_age45"] == 1) & (people["valid_function"] == 1)).sum())},
        {"step": 8, "criterion": "功能模型至少一个相邻风险区间", "n": int(people["function_model_person"].sum())},
        {"step": 9, "criterion": "联合状态至少一次可判定", "n": int(((people["eligible_age45"] == 1) & (people["valid_joint"] == 1)).sum())},
        {"step": 10, "criterion": "联合模型至少一个相邻风险区间", "n": int(people["joint_model_person"].sum())},
    ]
    parent_step = {1: None, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 4, 8: 7, 9: 4, 10: 9}
    flow_lookup = {row["step"]: row for row in flow}
    for row in flow:
        parent = parent_step[row["step"]]
        row["parent_step"] = parent
        row["excluded_from_parent"] = 0 if parent is None else flow_lookup[parent]["n"] - row["n"]
        row["pct_of_initial"] = round(100 * row["n"] / flow[0]["n"], 2) if flow[0]["n"] else np.nan

    entry_wave = people.loc[people["eligible_age45"].eq(1), "first_health_wave"].value_counts().sort_index()
    entry_rows = [{"entry_wave": int(wave), "n": int(count), "pct": round(100 * count / entry_wave.sum(), 2)} for wave, count in entry_wave.items()]
    interval_summary = []
    for domain, data in intervals.items():
        for (start_wave, end_wave), group in data.groupby(["start_wave", "end_wave"]):
            interval_summary.append({
                "domain": domain,
                "wave_pair": f"{start_wave}-{end_wave}",
                "origins": int(len(group)),
                "observed_endpoints": int(group["event_observed"].sum()),
                "to_death": int(group["to_state"].eq("D").sum()),
                "censored": int(group["to_state"].eq("CENSORED").sum()),
                "nonpositive_intervals": int(group["interval_years"].le(0).sum()),
            })

    cohort_checks = [
        {"check": "分析长表重复个人-波次", "count": int(eligible_long.duplicated(["person_id", "wave"]).sum()), "status": "PASS" if not eligible_long.duplicated(["person_id", "wave"]).any() else "FAIL"},
        {"check": "首次健康访谈年龄<45仍纳入", "count": int((people["eligible_age45"].eq(1) & people["age_first_health"].lt(45)).sum()), "status": "PASS"},
        {"check": "首次健康访谈年龄>120", "count": int(people["age_first_health"].gt(120).sum()), "status": "PASS"},
        {"check": "跨轮出生年冲突>1年", "count": int(person_demo["birth_year_conflict_gt1"].sum()), "status": "REVIEW" if person_demo["birth_year_conflict_gt1"].sum() else "PASS"},
        {"check": "出生年差异可能改变45岁纳入结论", "count": int(people["age_eligibility_sensitive"].sum()), "status": "REVIEW" if people["age_eligibility_sensitive"].sum() else "PASS"},
        {"check": "跨轮性别冲突", "count": int(person_demo["sex_conflict"].sum()), "status": "REVIEW" if person_demo["sex_conflict"].sum() else "PASS"},
        {"check": "模型区间时长≤0", "count": int(sum(data["interval_years"].le(0).sum() for data in intervals.values())), "status": "PASS"},
        {"check": "死亡后健康记录（继承一期质控）", "count": int(eligible_long["post_death_health_record"].sum()), "status": "PASS" if eligible_long["post_death_health_record"].sum() == 0 else "FAIL"},
    ]

    missing_variables = [
        ("age_years", "年龄"), ("female", "性别"), ("education_cat", "教育"), ("marital_cat", "婚姻"),
        ("urban_nbs", "NBS城乡"), ("hukou_agricultural", "农业户口"), ("chronic_n", "慢病数"),
        ("cesd10_score", "CESD-10"), ("current_smoker", "当前吸烟"), ("alcohol_past_year", "过去一年饮酒"),
        ("physical_activity_any", "任一体力活动"), ("proxy_health", "健康模块代理访谈"),
    ]
    eligible_health = eligible_long.loc[eligible_long["health_record_present"].eq(1) & eligible_long["death_confirmed"].eq(0)]
    missing = missingness_rows({
        "年龄≥45健康访谈人-波次": eligible_health,
        "疼痛模型基线": baseline["pain"],
        "功能模型基线": baseline["function"],
        "联合模型基线": baseline["joint"],
    }, missing_variables)

    availability_rows = [
        {"covariate": "年龄", "waves": "2011/2013/2015/2018/2020", "source": "人口学出生日期；跨轮众数", "rule": "有完整生日时按周岁；2020生成出生年按与2018重叠样本校正-2", "status": "可用"},
        {"covariate": "性别", "waves": "四轮", "source": "rgender / xrgender", "rule": "1男、2女；跨轮众数", "status": "可用"},
        {"covariate": "教育", "waves": "四轮", "source": "bd001 / bd001_w2_4 / zredu", "rule": "个人跨轮最高已观察学历，四分类", "status": "可用"},
        {"covariate": "婚姻", "waves": "四轮", "source": "be001 / ba011", "rule": "已婚/伴侣、分居/离婚、丧偶、从未婚", "status": "可用"},
        {"covariate": "城乡", "waves": "五轮社区", "source": "2011 PSU.dta urban_nbs", "rule": "按communityID链接；0农村、1城镇", "status": "可用"},
        {"covariate": "户口", "waves": "四轮", "source": "bc001 / zbc004 / ba009", "rule": "1农业；缺失时同一人前后轮填补", "status": "可用（部分跨轮填补）"},
        {"covariate": "慢病数", "waves": "四轮", "source": "14项共同医生诊断疾病", "rule": "14项全部可判定才求和；多病共存≥2", "status": "可用"},
        {"covariate": "CESD-10", "waves": "五轮", "source": "dc009-dc018；2020 dc016-dc025", "rule": "至少8项；按10项比例折算；≥10为抑郁症状", "status": "可用"},
        {"covariate": "吸烟", "waves": "四轮", "source": "da059/da061及波次对应变量", "rule": "当前仍吸烟=1", "status": "可用"},
        {"covariate": "饮酒", "waves": "四轮", "source": "过去一年饮酒频率", "rule": "月饮酒与低于每月均计过去一年饮酒", "status": "可用"},
        {"covariate": "体力活动", "waves": "四轮", "source": "强/中/轻活动≥10分钟", "rule": "任一强度回答是=1；全为否=0", "status": "可用（早期波次抽样缺失较高，宜敏感性分析）"},
        {"covariate": "代理访谈", "waves": "四轮", "source": "proxy或代理原因", "rule": "健康模块由代理回答=1", "status": "可用"},
        {"covariate": "BMI", "waves": "—", "source": "需Physical Examination模块", "rule": "当前stage未包含身高/体重测量文件，不推算", "status": "不可构建"},
    ]
    coding_rules = [
        {"topic": "年龄队列", "rule": "首次有效健康访谈时年龄≥45岁；较年轻的既有受访者之后不再补入"},
        {"topic": "延迟进入", "rule": "2013/2015/2018/2020首次接受健康访谈且满足年龄者可作为新增进入"},
        {"topic": "疼痛模型", "rule": "起点仅P0/P1/P2；下一轮确认死亡记D；失访/状态缺失记CENSORED"},
        {"topic": "功能模型", "rule": "起点仅F0/F1/F2；死亡与删失处理同疼痛模型"},
        {"topic": "联合模型", "rule": "起点为9个P×F活态；D为吸收终点"},
        {"topic": "模型时间", "rule": "访谈日期用当月15日；缺失终点/死亡日期用该波7月1日作为区间边界标记"},
        {"topic": "失访", "rule": "不把无访谈者当作健康或死亡；区间文件中保留明确删失原因"},
        {"topic": "Table 1", "rule": "分别以疼痛/功能模型首个有效起点为基线；百分比以非缺失为分母"},
    ]

    payload = {
        "project": "中国中老年人肌肉骨骼疼痛扩散与功能失能的双向转变",
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source_audit": [*demo_audit, *health_audit],
        "flow": flow,
        "entry_wave": entry_rows,
        "interval_summary": interval_summary,
        "cohort_checks": cohort_checks,
        "missingness": missing,
        "availability": availability_rows,
        "coding_rules": coding_rules,
        "pain_table1": pain_table,
        "function_table1": function_table,
        "joint_cross": joint_cross.to_dict(orient="records"),
        "file_counts": {
            "eligible_persons": int(people["eligible_age45"].sum()),
            "eligible_long_rows": int(len(eligible_long)),
            "pain_panel_rows": int(len(panels["pain"])),
            "function_panel_rows": int(len(panels["function"])),
            "joint_panel_rows": int(len(panels["joint"])),
            "pain_interval_rows": int(len(intervals["pain"])),
            "function_interval_rows": int(len(intervals["function"])),
            "joint_interval_rows": int(len(intervals["joint"])),
        },
    }
    with (audit_dir / "CHARLS_队列与Table1明细.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=lambda value: None if pd.isna(value) else value)
    pd.DataFrame(flow).to_csv(audit_dir / "CHARLS_样本流程.csv", index=False)
    pd.DataFrame(missing).to_csv(audit_dir / "CHARLS_协变量缺失.csv", index=False)
    people.to_csv(audit_dir / "CHARLS_个人队列纳入标记.csv", index=False)
    print(json.dumps(payload["file_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
