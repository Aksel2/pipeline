import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from sklearn.tree import DecisionTreeClassifier

from model_evaluate.evaluate import (
    calculate_metrics,
    evaluate_model,
    evaluate_all_models,
    save_results_to_csv,
    extract_model_rules,
    extract_decision_tree_rules,
    run_evaluation,
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


class TestEvaluateModel:
    def setup_method(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'f1': np.random.rand(40),
            'f2': np.random.rand(40)
        })
        self.y_train = pd.Series([0] * 20 + [1] * 20)
        self.X_test = pd.DataFrame({
            'f1': np.random.rand(10),
            'f2': np.random.rand(10)
        })
        self.y_test = pd.Series([0] * 5 + [1] * 5)
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_returns_expected_keys(self):
        result = evaluate_model(
            self.model, self.X_train, self.X_test,
            self.y_train, self.y_test, "test_model"
        )

        expected_keys = [
            'model_name', 'train_metrics', 'test_metrics',
            'train_confusion_matrix', 'test_confusion_matrix',
            'classification_report', 'y_test', 'y_test_pred', 'y_test_proba'
        ]
        for key in expected_keys:
            assert key in result

    def test_model_name_preserved(self):
        result = evaluate_model(
            self.model, self.X_train, self.X_test,
            self.y_train, self.y_test, "my_model"
        )
        assert result['model_name'] == "my_model"

    def test_handles_model_without_predict_proba(self):
        model = MagicMock()
        model.predict.return_value = np.array([0] * 5 + [1] * 5)
        del model.predict_proba

        result = evaluate_model(
            model, self.X_train, self.X_test,
            self.y_train, self.y_test, "no_proba_model"
        )

        assert result['y_test_proba'] is None


class TestEvaluateAllModels:
    def test_evaluates_multiple(self):
        np.random.seed(42)
        X_train = pd.DataFrame({'f1': np.random.rand(40)})
        y_train = pd.Series([0] * 20 + [1] * 20)
        X_test = pd.DataFrame({'f1': np.random.rand(10)})
        y_test = pd.Series([0] * 5 + [1] * 5)

        model1 = DecisionTreeClassifier(random_state=42)
        model1.fit(X_train, y_train)
        model2 = DecisionTreeClassifier(random_state=0, max_depth=1)
        model2.fit(X_train, y_train)

        trained_models = {
            'model_a': {'model': model1, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
            'model_b': {'model': model2, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
        }

        results = evaluate_all_models(trained_models)

        assert 'model_a' in results
        assert 'model_b' in results


class TestSaveResultsToCsv:
    def test_creates_file(self, tmp_path):
        all_results = {
            'model_a': {
                'train_metrics': {'accuracy': 0.9},
                'test_metrics': {'accuracy': 0.8},
            }
        }

        df = save_results_to_csv(all_results, str(tmp_path))

        assert os.path.exists(os.path.join(str(tmp_path), 'model_comparison.csv'))
        assert 'model' in df.columns
        assert 'train_accuracy' in df.columns
        assert 'test_accuracy' in df.columns


class TestExtractModelRules:
    def test_decision_tree_dispatch(self):
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.rand(20), 'f2': np.random.rand(20)})
        y = pd.Series([0] * 10 + [1] * 10)
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        rules = extract_model_rules(model, 'decision_tree', ['f1', 'f2'])
        assert 'f1' in rules or 'f2' in rules

    def test_unknown_model_type(self):
        model = MagicMock()
        rules = extract_model_rules(model, 'unknown_model')
        assert "not implemented" in rules

    def test_ripper_dispatch(self):
        model = MagicMock()
        model.__str__ = lambda self: "rule1 ^ rule2"
        rules = extract_model_rules(model, 'ripper')
        assert "rule1" in rules

    def test_figs_dispatch(self):
        model = MagicMock()
        model.__str__ = lambda self: "figs_rule"
        rules = extract_model_rules(model, 'figs')
        assert "figs_rule" in rules

    def test_c45_dispatch(self):
        model = MagicMock()
        model.__str__ = lambda self: "c45_tree"
        rules = extract_model_rules(model, 'c45')
        assert "c45_tree" in rules


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
