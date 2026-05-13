from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from decision_pipeline.model_evaluate.understandability import (
    calculate_ruleset_understandability,
    calculate_tree_understandability,
)


def calculate_metrics(y_true, y_pred, y_pred_proba=None, metrics_list=None):
    if metrics_list is None:
        metrics_list = ["accuracy", "precision", "recall", "f1", "auroc"]

    results = {}
    metrics_list = [m.lower() for m in metrics_list]

    if "accuracy" in metrics_list:
        results["accuracy"] = accuracy_score(y_true, y_pred)

    if "precision" in metrics_list:
        results["precision"] = precision_score(y_true, y_pred, average='binary', zero_division=0)

    if "recall" in metrics_list:
        results["recall"] = recall_score(y_true, y_pred, average='binary', zero_division=0)

    if "f1" in metrics_list:
        results["f1"] = f1_score(y_true, y_pred, average='binary', zero_division=0)

    if "auroc" in metrics_list:
        if y_pred_proba is not None:
            try:
                results["auroc"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                results["auroc"] = None
        else:
            results["auroc"] = None

    return results


def predict_with_proba(model, X):
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError, ValueError):
            pass
    return y_pred, y_proba


def evaluate_model(model, X_test, y_test, model_name, metrics_list=None, tree_kwargs=None, ruleset_kwargs=None):
    y_test_pred, y_test_proba = predict_with_proba(model, X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, metrics_list)

    tree_kwargs = tree_kwargs or {}
    ruleset_kwargs = ruleset_kwargs or {}
    if model_name in {'ripper', 'rulefit', 'skope_rules'}:
        understandability_result = calculate_ruleset_understandability(model, model_name, **ruleset_kwargs)
    else:
        understandability_result = calculate_tree_understandability(model, model_name, **tree_kwargs)

    return {
        'model_name': model_name,
        'test_metrics': test_metrics,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'understandability': understandability_result
    }


def evaluate_all_models(trained_models, metrics_list=None, tree_kwargs=None, ruleset_kwargs=None):
    all_results = {}

    for model_name, model_data in trained_models.items():
        print(f"\nEvaluating {model_name}...")

        results = evaluate_model(
            model_data['model'],
            model_data['X_test'],
            model_data['y_test'],
            model_name,
            metrics_list,
            tree_kwargs,
            ruleset_kwargs,
        )
        all_results[model_name] = results

    return all_results
