import pytest
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from model_evaluate.evaluate import (
    calculate_metrics,
    run_evaluation,
    _extract_tree_metrics,
    calculate_tree_understandability,
    _extract_ruleset_metrics,
)


class TestCalculateMetrics:
    def test_all_correct(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        result = calculate_metrics(y_true, y_pred)

        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        result = calculate_metrics(y_true, y_pred)

        assert result["accuracy"] == 0.0

    def test_auroc_with_proba(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.8, 0.9]
        result = calculate_metrics(y_true, y_pred, y_proba)

        assert result["auroc"] == 1.0

    def test_auroc_without_proba(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        result = calculate_metrics(y_true, y_pred, metrics_list=["auroc"])

        assert result["auroc"] is None

    def test_specific_metrics_only(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        result = calculate_metrics(y_true, y_pred, metrics_list=["accuracy"])

        assert "accuracy" in result
        assert "precision" not in result


class TestExtractTreeMetrics:
    def test_simple_tree(self):
        """Build a small tree with known structure and verify N, D, DD, F."""
        X = pd.DataFrame({
            'f1': [0, 0, 1, 1, 0, 0, 1, 1],
            'f2': [0, 1, 0, 1, 0, 1, 0, 1],
        })
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        metrics = _extract_tree_metrics(model.tree_)

        assert metrics["N"] >= 1
        assert metrics["D"] > 0
        assert metrics["F"] >= 1
        assert isinstance(metrics["DD"], (int, float))

    def test_deeper_tree(self):
        """A deeper tree should have more nodes and higher D."""
        np.random.seed(42)
        X = pd.DataFrame({
            'f1': np.random.rand(100),
            'f2': np.random.rand(100),
            'f3': np.random.rand(100),
        })
        y = pd.Series((X['f1'] > 0.5).astype(int) ^ (X['f2'] > 0.3).astype(int))
        model = DecisionTreeClassifier(random_state=42, max_depth=4)
        model.fit(X, y)

        metrics = _extract_tree_metrics(model.tree_)

        assert metrics["N"] > 1
        assert metrics["D"] > 1
        assert metrics["F"] >= 2

    def test_single_node_tree(self):
        """A tree with only a root (pure data) should have N=0, D=0."""
        X = pd.DataFrame({'f1': [1, 2, 3, 4]})
        y = pd.Series([0, 0, 0, 0])
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        metrics = _extract_tree_metrics(model.tree_)

        assert metrics["N"] == 0
        assert metrics["D"] == 0
        assert metrics["F"] == 0
        assert metrics["DD"] == 0


class TestCalculateTreeUnderstandability:
    def test_returns_dict_for_tree(self):
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.rand(20), 'f2': np.random.rand(20)})
        y = pd.Series([0] * 10 + [1] * 10)
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        result = calculate_tree_understandability(model, 'decision_tree')

        assert result is not None
        assert 'understandability' in result
        assert 0 <= result['understandability'] <= 1
        assert result['N'] >= 1
        assert result['F'] >= 1

    def test_score_decreases_with_complexity(self):
        """A deeper/more complex tree should have lower understandability."""
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.rand(100), 'f2': np.random.rand(100)})
        y = pd.Series([0] * 50 + [1] * 50)

        simple = DecisionTreeClassifier(max_depth=1, random_state=42)
        simple.fit(X, y)
        complex_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        complex_model.fit(X, y)

        simple_score = calculate_tree_understandability(simple, 'decision_tree')
        complex_score = calculate_tree_understandability(complex_model, 'decision_tree')

        assert simple_score['understandability'] >= complex_score['understandability']

    def test_s_parameter_affects_score(self):
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.rand(20), 'f2': np.random.rand(20)})
        y = pd.Series([0] * 10 + [1] * 10)
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        small_s = calculate_tree_understandability(model, 'decision_tree', s=1)
        large_s = calculate_tree_understandability(model, 'decision_tree', s=100)

        assert large_s['understandability'] >= small_s['understandability']


class TestExtractRulesetMetrics:
    def test_example_from_spec(self):
        """A & B -> 0, C & A -> 1, D -> 0. Flattened: [A, B, C, A, D]."""
        rules = [["A", "B"], ["C", "A"], ["D"]]
        metrics = _extract_ruleset_metrics(rules)

        assert metrics["N"] == 5
        assert metrics["D"] == pytest.approx(5 / 3)
        assert metrics["F"] == 4
        assert metrics["DD"] == 3

    def test_single_rule(self):
        rules = [["X", "Y", "Z"]]
        metrics = _extract_ruleset_metrics(rules)

        assert metrics["N"] == 3
        assert metrics["D"] == 3.0
        assert metrics["F"] == 3
        assert metrics["DD"] == 0

    def test_repeated_feature_three_times(self):
        """A appears at positions 0, 2, 4. DD(A) = (2-0) + (4-2) = 4."""
        rules = [["A", "B"], ["A", "C"], ["A"]]
        metrics = _extract_ruleset_metrics(rules)

        assert metrics["N"] == 5
        assert metrics["DD"] == 4
        assert metrics["F"] == 3

    def test_empty_returns_none(self):
        assert _extract_ruleset_metrics([]) is None
        assert _extract_ruleset_metrics(None) is None


class TestRunEvaluation:
    def test_end_to_end(self, tmp_path):
        np.random.seed(42)
        X_train = pd.DataFrame({'f1': np.random.rand(40)})
        y_train = pd.Series([0] * 20 + [1] * 20)
        X_test = pd.DataFrame({'f1': np.random.rand(10)})
        y_test = pd.Series([0] * 5 + [1] * 5)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        trained_models = {
            'decision_tree': {
                'model': model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        }

        config = {
            "evaluation_config": {
                "metrics": ["accuracy", "precision"],
                "output_directory": str(tmp_path)
            }
        }

        all_results, comparison_df = run_evaluation(trained_models, config)

        assert 'decision_tree' in all_results
        assert os.path.exists(os.path.join(str(tmp_path), 'model_comparison.csv'))
        assert os.path.exists(os.path.join(str(tmp_path), 'classification_reports.txt'))
