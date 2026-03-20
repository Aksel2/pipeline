import pandas as pd
import pytest

from preprocessing.preprocess import preprocess_event_log


def make_config():
    return {
        "preprocess_config": {
            "pre_decision_activities": ["A"],
            "post_decision_0_activities": ["B"],
            "post_decision_1_activities": ["C"],
            "column_names": {
                "case_id": "case_id",
                "activity": "activity",
                "start_time": "start_time",
                "end_time": "end_time"
            }
        },
        "encoding_config": {
            "case_attributes": [],
            "event_attributes_continuous": [],
            "event_attributes_discrete": []
        }
    }


def make_df(case_id, activities):
    rows = []
    for i, act in enumerate(activities):
        rows.append({
            "case_id": case_id,
            "activity": act,
            "start_time": f"2024-01-01 {i:02d}:00:00",
            "end_time": f"2024-01-01 {i:02d}:30:00"
        })
    return pd.DataFrame(rows)


class TestPreprocessEventLog:

    def test_azzc_scan_forward_finds_c(self):
        """AZZC: A is decision point, scan past Z's, find C → observation includes A, target=1."""
        df = make_df("case1", ["A", "Z", "Z", "C"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["activity"] == "A"
        assert result.iloc[0]["target"] == 1
        assert result.iloc[0]["observation"] == 1

    def test_azzab_overwrite(self):
        """AZZAB: Two A's. First A→B creates obs, second A→B overwrites.
        Previous observation is discarded. New observation starts after the previous decision point."""
        df = make_df("case1", ["A", "Z", "Z", "A", "B"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 3
        activities = result["activity"].tolist()
        assert activities == ["Z", "Z", "A"]
        assert all(result["target"] == 0)
        assert all(result["observation"] == 2)

    def test_azzz_no_outcome(self):
        """AZZZ: A is found but no B or C follows → not added."""
        df = make_df("case1", ["A", "Z", "Z", "Z"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 0

    def test_abcde_immediate_outcome(self):
        """ABCDE: A followed immediately by B → observation is A, target=0."""
        df = make_df("case1", ["A", "B", "C", "D", "E"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["activity"] == "A"
        assert result.iloc[0]["target"] == 0

    def test_abghz(self):
        """ABGHZ: A followed by B → observation is A, target=0."""
        df = make_df("case1", ["A", "B", "G", "H", "Z"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["activity"] == "A"
        assert result.iloc[0]["target"] == 0

    def test_abcbfb(self):
        """ABCBFB: A followed by B → observation is A, target=0."""
        df = make_df("case1", ["A", "B", "C", "B", "F", "B"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["activity"] == "A"
        assert result.iloc[0]["target"] == 0

    def test_abghc_takes_first_outcome(self):
        """ABGHC: A followed by B (first outcome found) → target=0, not C."""
        df = make_df("case1", ["A", "B", "G", "H", "C"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["target"] == 0

    def test_abcf_takes_b(self):
        """ABCF: A followed by B → target=0 (we take B)."""
        df = make_df("case1", ["A", "B", "C", "F"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["target"] == 0

    def test_no_decision_point(self):
        """ZZZZZ: No A at all → nothing added."""
        df = make_df("case1", ["Z", "Z", "Z", "Z", "Z"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 0

    def test_multiple_cases(self):
        """Two cases: one valid, one not."""
        df1 = make_df("case1", ["A", "B"])
        df2 = make_df("case2", ["Z", "Z"])
        df = pd.concat([df1, df2], ignore_index=True)
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["case_id"] == "case1"

    def test_observation_includes_all_prior_rows(self):
        """XYZAC: Activities before A are included in the observation."""
        df = make_df("case1", ["X", "Y", "Z", "A", "C"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 4
        activities = result["activity"].tolist()
        assert activities == ["X", "Y", "Z", "A"]
        assert all(result["target"] == 1)
        assert all(result["observation"] == 1)
