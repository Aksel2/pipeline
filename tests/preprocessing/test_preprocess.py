import pandas as pd

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

    def test_azzc(self):
        """AZZC: A sets pending, C is outcome -> observation is A (up to pre-decision only)."""
        df = make_df("case1", ["A", "Z", "Z", "C"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result["activity"].tolist() == ["A"]
        assert all(result["target"] == 1)
        assert all(result["observation"] == 1)

    def test_azzab(self):
        """AZZAB: Two A's, second replaces first. B is outcome.
        Observation is everything before B: A,Z,Z,A."""
        df = make_df("case1", ["A", "Z", "Z", "A", "B"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 4
        assert result["activity"].tolist() == ["A", "Z", "Z", "A"]
        assert all(result["target"] == 0)
        assert all(result["observation"] == 1)

    def test_azzz_no_outcome(self):
        """AZZZ: A is found but no B or C follows -> no observation."""
        df = make_df("case1", ["A", "Z", "Z", "Z"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 0

    def test_ab_immediate_outcome(self):
        """AB: A sets pending, B is immediate outcome -> observation is just A."""
        df = make_df("case1", ["A", "B"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["activity"] == "A"
        assert result.iloc[0]["target"] == 0

    def test_abghz(self):
        """ABGHZ: A then B -> observation is A, target=0. Events after B ignored."""
        df = make_df("case1", ["A", "B", "G", "H", "Z"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["activity"] == "A"
        assert result.iloc[0]["target"] == 0

    def test_no_decision_point(self):
        """ZZZZZ: No A at all -> no observation."""
        df = make_df("case1", ["Z", "Z", "Z", "Z", "Z"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 0

    def test_multiple_cases(self):
        """Two cases: one with decision point, one without."""
        df1 = make_df("case1", ["A", "B"])
        df2 = make_df("case2", ["Z", "Z"])
        df = pd.concat([df1, df2], ignore_index=True)
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["case_id"] == "case1"

    def test_observation_includes_all_prior_rows(self):
        """XYZAC: Events before A are included. Observation = X,Y,Z,A (everything before C)."""
        df = make_df("case1", ["X", "Y", "Z", "A", "C"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 4
        assert result["activity"].tolist() == ["X", "Y", "Z", "A"]
        assert all(result["target"] == 1)
        assert all(result["observation"] == 1)

    def test_two_observations(self):
        """AZZBAZZB: Two complete decision cycles.
        Obs 1: A (up to pre-decision). Obs 2: A,Z,Z,B,A (full history up to second pre-decision)."""
        df = make_df("case1", ["A", "Z", "Z", "B", "A", "Z", "Z", "B"])
        result = preprocess_event_log(df, make_config())

        obs1 = result[result["observation"] == 1]
        obs2 = result[result["observation"] == 2]

        assert len(obs1) == 1
        assert obs1["activity"].tolist() == ["A"]
        assert all(obs1["target"] == 0)

        assert len(obs2) == 5
        assert obs2["activity"].tolist() == ["A", "Z", "Z", "B", "A"]
        assert all(obs2["target"] == 0)

    def test_outcome_without_decision_ignored(self):
        """BZZ: Outcome B appears but no prior A -> no observation."""
        df = make_df("case1", ["B", "Z", "Z"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 0

    def test_pending_reset_after_outcome(self):
        """ABZC: A->B commits obs 1. Z,C has no pending A -> no second observation."""
        df = make_df("case1", ["A", "B", "Z", "C"])
        result = preprocess_event_log(df, make_config())

        assert len(result) == 1
        assert result.iloc[0]["activity"] == "A"
        assert result.iloc[0]["target"] == 0
