import pytest
import os
import math
import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.tree import DecisionTreeClassifier

from decision_pipeline.model_evaluate.evaluate import run_evaluation
from decision_pipeline.model_evaluate.metrics import calculate_metrics
from decision_pipeline.model_evaluate.understandability import (
    calculate_tree_understandability,
    compute_understandability_score,
    extract_ruleset_metrics,
    extract_tree_metrics,
)


class TestCalculateMetrics:
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
        assert "auroc" not in result


class TestExactTreeMetrics:
    @pytest.fixture
    def mock_tree(self):
        tree = SimpleNamespace()
        tree.children_left = np.array([1, 2, -1, -1, 5, -1, 7, -1, -1])
        tree.children_right = np.array([4, 3, -1, -1, 6, -1, 8, -1, -1])
        tree.feature = np.array([0, 1, -2, -2, 2, -2, 0, -2, -2])
        return tree

    def test_N(self, mock_tree):
        assert extract_tree_metrics(mock_tree)["N"] == 4

    def test_D(self, mock_tree):
        assert extract_tree_metrics(mock_tree)["D"] == pytest.approx(2.4)

    def test_DD(self, mock_tree):
        assert extract_tree_metrics(mock_tree)["DD"] == 2

    def test_F(self, mock_tree):
        assert extract_tree_metrics(mock_tree)["F"] == 3

    def test_understandability_score(self, mock_tree):
        metrics = extract_tree_metrics(mock_tree)
        result = compute_understandability_score(
            metrics["N"], metrics["D"], metrics["DD"], metrics["F"],
            s=28, w1=1, w2=1, w3=1,
        )
        c = 1 * (4 + 2.4) + 1 * 2 + 1 * 3
        expected = math.exp(-((c / 28) ** 2))
        assert result["understandability"] == pytest.approx(expected)
        assert result["x"] == pytest.approx(11.4)


class TestBranchAwareDD:
    def test_cross_branch_repetition_contributes_zero(self):
        """Feature B at depth 1 (left branch) and depth 2 (right branch via C).
        Same depths in different branches must contribute 0 to DD, even though
        max-min across all occurrences would be 1."""
        tree = SimpleNamespace()
        tree.children_left  = np.array([1, 3, 5, -1, -1, -1, 7, -1, -1])
        tree.children_right = np.array([2, 4, 6, -1, -1, -1, 8, -1, -1])
        tree.feature        = np.array([0, 1, 2, -2, -2, -2, 1, -2, -2])

        result = extract_tree_metrics(tree)
        assert result["DD"] == 0
        assert result["F"] == 3

    def test_same_branch_repetition_contributes(self):
        """Feature 0 at depth 0 (root) and depth 2 (under root's right via
        feature 2). Same root-to-leaf path, so DD must include spread = 2."""
        tree = SimpleNamespace()
        tree.children_left  = np.array([1, 2, -1, -1, 5, -1, 7, -1, -1])
        tree.children_right = np.array([4, 3, -1, -1, 6, -1, 8, -1, -1])
        tree.feature        = np.array([0, 1, -2, -2, 2, -2, 0, -2, -2])

        result = extract_tree_metrics(tree)
        assert result["DD"] == 2


class TestCalculateTreeUnderstandability:
    def test_score_decreases_with_complexity(self):
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
    def test_table1_exact(self):
        rules = [
            ["credit_score", "loan_amount"],
            ["credit_score", "loan_amount"],
            ["credit_score", "income", "credit_score"],
            ["credit_score", "income"],
            ["credit_score", "income"],
        ]
        metrics = extract_ruleset_metrics(rules)

        assert metrics["N"] == 11
        assert metrics["D"] == pytest.approx(11 / 5)
        assert metrics["DD"] == 16
        assert metrics["F"] == 3

    def test_example_from_spec(self):
        """A & B -> 0, C & A -> 1, D -> 0. Flattened: [A, B, C, A, D]."""
        rules = [["A", "B"], ["C", "A"], ["D"]]
        metrics = extract_ruleset_metrics(rules)

        assert metrics["N"] == 5
        assert metrics["D"] == pytest.approx(5 / 3)
        assert metrics["F"] == 4
        assert metrics["DD"] == 3

    def test_single_rule(self):
        rules = [["X", "Y", "Z"]]
        metrics = extract_ruleset_metrics(rules)

        assert metrics["N"] == 3
        assert metrics["D"] == 3.0
        assert metrics["F"] == 3
        assert metrics["DD"] == 0

    def test_repeated_feature_three_times(self):
        """A appears at positions 0, 2, 4. DD(A) = (2-0) + (4-2) = 4."""
        rules = [["A", "B"], ["A", "C"], ["A"]]
        metrics = extract_ruleset_metrics(rules)

        assert metrics["N"] == 5
        assert metrics["DD"] == 4
        assert metrics["F"] == 3

    def test_empty_returns_none(self):
        assert extract_ruleset_metrics([]) is None
        assert extract_ruleset_metrics(None) is None


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
                "metrics": ["accuracy", "auroc"],
                "output_directory": str(tmp_path)
            }
        }

        all_results, comparison_df = run_evaluation(trained_models, config)

        assert 'decision_tree' in all_results
        assert os.path.exists(os.path.join(str(tmp_path), 'model_comparison.csv'))
        assert os.path.exists(os.path.join(str(tmp_path), 'classification_reports.txt'))
