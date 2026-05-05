import pytest
import json
import pandas as pd

from core import pipeline


def make_event_log():
    rows = []
    for case_id in range(1, 21):
        outcome_activity = "Approve application" if case_id % 2 == 0 else "Reject application"
        events = [
            (case_id, "Submit application", "2024-01-01 08:00", "2024-01-01 08:30", "Urgent", "Standard", 10000.0),
            (case_id, "Check documents", "2024-01-01 09:00", "2024-01-01 09:30", "Urgent", "Standard", 10000.0),
            (case_id, "Assess loan risk", "2024-01-01 10:00", "2024-01-01 10:30", "Urgent", "Standard", 10000.0),
            (case_id, outcome_activity, "2024-01-01 11:00", "2024-01-01 11:30", "Urgent", "Standard", 10000.0),
        ]
        for e in events:
            rows.append({
                'case_id': e[0], 'activity': e[1],
                'start_time': e[2], 'end_time': e[3],
                'Loan_type': e[4], 'Client_type': e[5], 'Loan_amount': e[6]
            })
    return pd.DataFrame(rows)


def make_config(output_dir):
    return {
        "preprocess_config": {
            "pre_decision_activities": ["Assess loan risk"],
            "post_decision_0_activities": ["Approve application"],
            "post_decision_1_activities": ["Reject application"],
            "column_names": {
                "case_id": "case_id",
                "activity": "activity",
                "start_time": "start_time",
                "end_time": "end_time"
            }
        },
        "encoding_config": {
            "case_attributes": ["Loan_type", "Client_type"],
            "event_attributes_continuous": ["Loan_amount"],
            "event_attributes_discrete": [],
            "encoding_strategies": {
                "one-hot": ["Client_type", "Loan_type"],
                "target-based": [],
                "aggregation": []
            }
        },
        "models_config": {
            "decision_tree_classifier": {
                "enabled": True,
                "max_depth": 3,
                "criterion": "gini",
                "class_weight": "balanced"
            },
            "figs_classifier": {"enabled": False},
            "ripper_classifier": {"enabled": False},
            "explainable_boosting_classifier": {"enabled": False}
        },
        "evaluation_config": {
            "metrics": ["accuracy", "precision"],
            "output_directory": str(output_dir)
        }
    }


class TestPipeline:
    def test_full_pipeline(self, tmp_path):
        data_path = str(tmp_path / "data.csv")
        config_path = str(tmp_path / "config.json")
        output_dir = tmp_path / "outputs"

        df = make_event_log()
        df.to_csv(data_path, index=False)

        config = make_config(str(output_dir))
        with open(config_path, 'w') as f:
            json.dump(config, f)

        train_df, test_df, trained_models, eval_results = pipeline(
            input_logs_path=data_path,
            config_path=config_path,
            test_percentage=0.2
        )

        assert len(train_df) > 0
        assert len(test_df) > 0
        assert 'decision_tree' in trained_models
        assert 'decision_tree' in eval_results

    def test_no_models_enabled(self, tmp_path):
        data_path = str(tmp_path / "data.csv")
        config_path = str(tmp_path / "config.json")

        df = make_event_log()
        df.to_csv(data_path, index=False)

        config = make_config(str(tmp_path / "outputs"))
        for model in config["models_config"]:
            config["models_config"][model]["enabled"] = False

        with open(config_path, 'w') as f:
            json.dump(config, f)

        train_df, test_df, trained_models, eval_results = pipeline(
            input_logs_path=data_path,
            config_path=config_path,
        )

        assert trained_models == {}
        assert eval_results == {}

    def test_separate_test_file(self, tmp_path):
        train_path = str(tmp_path / "train.csv")
        test_path = str(tmp_path / "test.csv")
        config_path = str(tmp_path / "config.json")

        df = make_event_log()
        df[df['case_id'] <= 14].to_csv(train_path, index=False)
        df[df['case_id'] > 14].to_csv(test_path, index=False)

        config = make_config(str(tmp_path / "outputs"))
        with open(config_path, 'w') as f:
            json.dump(config, f)

        train_df, test_df, trained_models, eval_results = pipeline(
            input_logs_path=train_path,
            test_logs_path=test_path,
            config_path=config_path,
        )

        assert len(train_df) > 0
        assert len(test_df) > 0
        assert 'decision_tree' in trained_models

    def test_missing_pre_decision_raises(self, tmp_path):
        data_path = str(tmp_path / "data.csv")
        config_path = str(tmp_path / "config.json")

        df = make_event_log()
        df.to_csv(data_path, index=False)

        config = make_config(str(tmp_path / "outputs"))
        config["preprocess_config"]["pre_decision_activities"] = []
        config["preprocess_config"].pop("bpmn_model_path", None)

        with open(config_path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="pre_decision_activity"):
            pipeline(input_logs_path=data_path, config_path=config_path)
