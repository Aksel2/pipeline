import pytest
import pandas as pd


from decision_pipeline.encoding.encoding import collapse_observations, DataFrameEncoder


def make_config(one_hot=None, target_based=None, aggregation=None,
                case_attributes=None, event_continuous=None, event_discrete=None):
    return {
        "encoding_config": {
            "case_attributes": case_attributes or [],
            "event_attributes_continuous": event_continuous or [],
            "event_attributes_discrete": event_discrete or [],
            "encoding_strategies": {
                "one-hot": one_hot or [],
                "target-based": target_based or [],
                "aggregation": aggregation or []
            }
        }
    }


class TestCollapseObservations:
    def test_last_state_default(self):
        df = pd.DataFrame({
            'case_id': [1, 1, 1],
            'activity': ['A', 'B', 'C'],
            'amount': [10, 20, 30],
            'observation': [1, 1, 1],
            'target': [0, 0, 0]
        })
        config = make_config(event_continuous=["amount"])
        result = collapse_observations(df, config)

        assert len(result) == 1
        assert result.iloc[0]['activity'] == 'C'
        assert result.iloc[0]['amount'] == 30

    def test_aggregation_sums(self):
        df = pd.DataFrame({
            'case_id': [1, 1, 1],
            'activity': ['A', 'B', 'C'],
            'duration': [10.0, 20.0, 30.0],
            'observation': [1, 1, 1],
            'target': [0, 0, 0]
        })
        config = make_config(
            event_continuous=["duration"],
            aggregation=["duration"]
        )
        result = collapse_observations(df, config)

        assert len(result) == 1
        assert result.iloc[0]['duration'] == 60.0
        assert result.iloc[0]['activity'] == 'C'

    def test_multiple_observations(self):
        df = pd.DataFrame({
            'case_id': [1, 1, 1, 1],
            'activity': ['A', 'B', 'C', 'D'],
            'amount': [10, 20, 30, 40],
            'observation': [1, 1, 2, 2],
            'target': [0, 0, 1, 1]
        })
        config = make_config(event_continuous=["amount"])
        result = collapse_observations(df, config)

        assert len(result) == 2
        obs1 = result[result['observation'] == 1].iloc[0]
        obs2 = result[result['observation'] == 2].iloc[0]
        assert obs1['amount'] == 20
        assert obs2['amount'] == 40

    def test_aggregation_rejects_case_attribute(self):
        df = pd.DataFrame({
            'case_id': [1], 'loan_type': ['A'],
            'observation': [1], 'target': [0]
        })
        config = make_config(
            case_attributes=["loan_type"],
            aggregation=["loan_type"]
        )
        with pytest.raises(ValueError, match=r".*case attribute.*"):
            collapse_observations(df, config)

    def test_aggregation_rejects_discrete_attribute(self):
        df = pd.DataFrame({
            'case_id': [1], 'status': ['open'],
            'observation': [1], 'target': [0]
        })
        config = make_config(
            event_discrete=["status"],
            aggregation=["status"]
        )
        with pytest.raises(ValueError, match=r".*discrete event attribute.*"):
            collapse_observations(df, config)

    def test_aggregation_rejects_unknown_column(self):
        df = pd.DataFrame({
            'case_id': [1], 'amount': [10],
            'observation': [1], 'target': [0]
        })
        config = make_config(aggregation=["amount"])
        with pytest.raises(ValueError, match=r".*not in event_attributes_continuous.*"):
            collapse_observations(df, config)

    def test_mixed_aggregation_and_last_state(self):
        df = pd.DataFrame({
            'case_id': [1, 1, 1],
            'activity': ['A', 'B', 'C'],
            'cost': [100.0, 200.0, 300.0],
            'status': ['new', 'pending', 'done'],
            'observation': [1, 1, 1],
            'target': [0, 0, 0]
        })
        config = make_config(
            event_continuous=["cost"],
            aggregation=["cost"]
        )
        result = collapse_observations(df, config)

        assert result.iloc[0]['cost'] == 600.0
        assert result.iloc[0]['status'] == 'done'


class TestDataFrameEncoderOneHot:
    def test_basic_one_hot(self):
        df = pd.DataFrame({
            'case_id': [1, 2, 3],
            'color': ['red', 'blue', 'green'],
            'value': [10, 20, 30],
            'target': [0, 1, 0]
        })
        config = make_config(
            case_attributes=["color"],
            one_hot=["color"]
        )
        encoder = DataFrameEncoder(config)
        result = encoder.fit_transform(df)

        assert 'color' not in result.columns
        onehot_cols = [c for c in result.columns if c.startswith('color_')]
        assert len(onehot_cols) == 2

    def test_transform_unseen_category(self):
        df_train = pd.DataFrame({
            'case_id': [1, 2],
            'type': ['A', 'B'],
            'target': [0, 1]
        })
        df_test = pd.DataFrame({
            'case_id': [3],
            'type': ['C'],
            'target': [0]
        })
        config = make_config(case_attributes=["type"], one_hot=["type"])
        encoder = DataFrameEncoder(config)
        encoder.fit(df_train)
        result = encoder.transform(df_test)

        onehot_cols = [c for c in result.columns if c.startswith('type_')]
        assert all(result[col].iloc[0] == 0.0 for col in onehot_cols)

    def test_missing_column_raises(self):
        df_train = pd.DataFrame({
            'case_id': [1], 'type': ['A'], 'target': [0]
        })
        df_test = pd.DataFrame({
            'case_id': [1], 'target': [0]
        })
        config = make_config(case_attributes=["type"], one_hot=["type"])
        encoder = DataFrameEncoder(config)
        encoder.fit(df_train)
        with pytest.raises(ValueError, match=r".*Columns missing.*"):
            encoder.transform(df_test)


class TestDataFrameEncoderTargetBased:
    def test_basic_target_encoding(self):
        df = pd.DataFrame({
            'case_id': [1, 2, 3, 4],
            'color': ['red', 'red', 'blue', 'blue'],
            'target': [1, 1, 0, 0]
        })
        config = make_config(
            case_attributes=["color"],
            target_based=["color"]
        )
        encoder = DataFrameEncoder(config)
        result = encoder.fit_transform(df)

        assert 'color' not in result.columns
        assert 'color_encoded' in result.columns
        red_vals = result.iloc[:2]['color_encoded'].values
        blue_vals = result.iloc[2:]['color_encoded'].values
        assert all(v == 1.0 for v in red_vals)
        assert all(v == 0.0 for v in blue_vals)

    def test_unknown_category_uses_global_mean(self):
        df_train = pd.DataFrame({
            'case_id': [1, 2, 3, 4],
            'color': ['red', 'red', 'blue', 'blue'],
            'target': [1, 1, 0, 0]
        })
        df_test = pd.DataFrame({
            'case_id': [5],
            'color': ['green'],
            'target': [0]
        })
        config = make_config(
            case_attributes=["color"],
            target_based=["color"]
        )
        encoder = DataFrameEncoder(config)
        encoder.fit(df_train)
        result = encoder.transform(df_test)

        assert result.iloc[0]['color_encoded'] == 0.5

class TestDataFrameEncoderMixed:
    def test_mixed_onehot_and_target(self):
        df = pd.DataFrame({
            'case_id': [1, 2, 3, 4],
            'type': ['A', 'B', 'A', 'B'],
            'color': ['red', 'red', 'blue', 'blue'],
            'value': [10, 20, 30, 40],
            'target': [1, 0, 1, 0]
        })
        config = make_config(
            case_attributes=["type", "color"],
            one_hot=["type"],
            target_based=["color"]
        )
        encoder = DataFrameEncoder(config)
        result = encoder.fit_transform(df)

        assert 'type' not in result.columns
        assert 'color' not in result.columns
        assert 'color_encoded' in result.columns
        assert any(c.startswith('type_') for c in result.columns)
        assert 'value' in result.columns
