import pytest
import pandas as pd
import numpy as np

from model_train.models import prepare_features_and_target


def make_sample_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        'case_id': range(n),
        'activity': ['A'] * n,
        'start_time': ['2024-01-01'] * n,
        'end_time': ['2024-01-02'] * n,
        'observation': range(n),
        'feature_1': np.random.rand(n),
        'feature_2': np.random.rand(n),
        'target': [0] * 25 + [1] * 25
    })


class TestPrepareFeatures:
    def test_drops_non_feature_columns(self):
        df = make_sample_df()
        X, y = prepare_features_and_target(df)

        for col in ['target', 'observation', 'case_id', 'activity', 'start_time', 'end_time']:
            assert col not in X.columns

    def test_keeps_feature_columns(self):
        df = make_sample_df()
        X, y = prepare_features_and_target(df)

        assert 'feature_1' in X.columns
        assert 'feature_2' in X.columns

    def test_returns_correct_target(self):
        df = make_sample_df()
        X, y = prepare_features_and_target(df)

        assert len(y) == 50
        assert y.dtype in [np.float64, np.int64, float, int]

    def test_filters_na_targets(self):
        df = make_sample_df()
        df.loc[0, 'target'] = pd.NA
        X, y = prepare_features_and_target(df)

        assert len(X) == 49
        assert len(y) == 49

    def test_handles_missing_optional_columns(self):
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0],
            'observation': [1, 2],
            'target': [0, 1]
        })
        X, y = prepare_features_and_target(df)

        assert 'feature_1' in X.columns
        assert len(X) == 2
