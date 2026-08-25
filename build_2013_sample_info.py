#!/usr/bin/env python3
"""Construct the 2013 wave-status file used by the five-wave pipeline.

The 2013 public release has no separate Sample_Infor.dta in the downloaded
bundle.  Interview year/month and the longitudinal-death flag are available in
Weights.dta, while Exit_Interview.dta provides direct evidence of death.  This
script creates a transparent, auditable wave-status file using their union.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "stage" / "2013"
AUDIT = ROOT / "phase1" / "2013_sample_construction_audit.json"


def clean_id(series: pd.Series) -> pd.Series:
    value = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return value.where(value.str.fullmatch(r"\d{12}").fillna(False))


def source_frame(path: Path, columns: list[str], suffix: str) -> pd.DataFrame:
    frame = pd.read_stata(path, columns=columns, convert_categoricals=False)
    frame["ID"] = clean_id(frame["ID"])
    frame = frame.dropna(subset=["ID"]).drop_duplicates("ID", keep="first")
    rename = {
        "householdID": f"householdID_{suffix}",
        "communityID": f"communityID_{suffix}",
    }
    return frame.rename(columns=rename)


def main() -> None:
    weights = source_frame(
        STAGE / "Weights.dta",
        ["ID", "householdID", "communityID", "iyear", "imonth", "INDV_L_Died"],
        "weight",
    )
    health = source_frame(
        STAGE / "Health_Status_and_Functioning.dta",
        ["ID", "householdID", "communityID"],
        "health",
    )
    exit_interview = source_frame(
        STAGE / "Exit_Interview.dta",
        ["ID", "householdID", "communityID", "exb001_1", "exb001_2"],
        "exit",
    )

    ids = pd.concat([weights[["ID"]], health[["ID"]], exit_interview[["ID"]]], ignore_index=True).drop_duplicates()
    frame = ids.merge(weights, on="ID", how="left", validate="one_to_one")
    frame = frame.merge(health, on="ID", how="left", validate="one_to_one")
    frame = frame.merge(exit_interview, on="ID", how="left", validate="one_to_one", indicator="exit_merge")

    for stem in ("householdID", "communityID"):
        frame[stem] = frame[f"{stem}_weight"].combine_first(frame[f"{stem}_health"]).combine_first(frame[f"{stem}_exit"])
    frame["iyear"] = pd.to_numeric(frame["iyear"], errors="coerce").where(lambda x: x.between(2013, 2014)).fillna(2013)
    frame["imonth"] = pd.to_numeric(frame["imonth"], errors="coerce").where(lambda x: x.between(1, 12)).fillna(7)
    weight_died = pd.to_numeric(frame["INDV_L_Died"], errors="coerce").eq(1)
    exit_died = frame["exit_merge"].eq("both")
    frame["died"] = (weight_died | exit_died).astype(int)

    output = frame[["ID", "householdID", "communityID", "iyear", "imonth", "died"]].copy()
    output.to_stata(STAGE / "Sample_Infor.dta", write_index=False, version=119)

    audit = {
        "rule": "died=1 if INDV_L_Died=1 in Weights.dta or an ID is present in Exit_Interview.dta",
        "union_person_ids": int(len(output)),
        "health_ids": int(health["ID"].nunique()),
        "weight_ids": int(weights["ID"].nunique()),
        "exit_interview_ids": int(exit_interview["ID"].nunique()),
        "weight_death_ids": int(weight_died.sum()),
        "exit_death_ids": int(exit_died.sum()),
        "union_confirmed_deaths": int(output["died"].sum()),
        "weight_and_exit_death_overlap": int((weight_died & exit_died).sum()),
        "health_death_conflicts": int((frame["ID"].isin(set(health["ID"])) & output["died"].eq(1)).sum()),
        "death_year_available": int(pd.to_numeric(frame["exb001_1"], errors="coerce").between(1900, 2014).sum()),
        "death_month_available": int(pd.to_numeric(frame["exb001_2"], errors="coerce").between(1, 12).sum()),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
