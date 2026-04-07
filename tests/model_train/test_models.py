import pytest
import pandas as pd
import numpy as np

from model_train.models import (
    prepare_features_and_target,
    train_dtc,
    train_c45,
    train_figs,
    train_ebc,
    train_ripper,
)


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


def make_train_test():
    df = make_sample_df()
    X, y = prepare_features_and_target(df)
    split = 40
    return X[:split], X[split:], y[:split], y[split:]


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


class TestTrainDtc:
    def test_returns_model(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_dtc(X_train, X_test, y_train, y_test)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'score')

    def test_accepts_params(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_dtc(X_train, X_test, y_train, y_test, {
            'max_depth': 3, 'criterion': 'entropy'
        })
        assert model.max_depth == 3

    def test_predictions_valid(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_dtc(X_train, X_test, y_train, y_test)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})


class TestTrainC45:
    def test_returns_model(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_c45(X_train, X_test, y_train, y_test)
        assert hasattr(model, 'predict')


class TestTrainFigs:
    def test_returns_model(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_figs(X_train, X_test, y_train, y_test)
        assert hasattr(model, 'predict')

    def test_accepts_max_rules(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_figs(X_train, X_test, y_train, y_test, {'max_rules': 5})
        assert hasattr(model, 'predict')


class TestTrainEbc:
    def test_returns_model(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_ebc(X_train, X_test, y_train, y_test)
        assert hasattr(model, 'predict')


class TestTrainRipper:
    def test_returns_model(self):
        X_train, X_test, y_train, y_test = make_train_test()
        model = train_ripper(X_train, X_test, y_train, y_test)
        assert hasattr(model, 'predict')
