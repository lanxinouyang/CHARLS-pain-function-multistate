from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fit_fivewave_ctmc as fit


class ExitInterviewSensitivityTests(unittest.TestCase):
    def test_weight_only_death_becomes_vital_unknown_not_alive(self) -> None:
        long = pd.DataFrame(
            [
                {
                    "person_id": "000000000001",
                    "wave": 2013,
                    "death_confirmed": 1,
                    "died_raw": 1,
                    "health_record_present": 0,
                    "sample_record_present": 1,
                    "vital_observation": "death_confirmed",
                    "pain_state_with_death": "D",
                    "function_state": "D",
                    "function_state_complete11": "D",
                    "joint_state": "D",
                }
            ]
        )

        recoded, mask = fit.apply_exit_interview_only_death_definition(long, set())

        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(pd.isna(recoded.loc[0, "died_raw"]))
        self.assertFalse(fit.legacy.valid_alive(recoded.loc[0]))
        self.assertEqual(recoded.loc[0, "vital_observation"], "unknown_after_exit_only_definition")


if __name__ == "__main__":
    unittest.main()
