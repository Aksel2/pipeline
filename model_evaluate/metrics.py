from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model_evaluate.understandability import (
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


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, metrics_list=None):
    y_train_pred, y_train_proba = predict_with_proba(model, X_train)
    y_test_pred, y_test_proba = predict_with_proba(model, X_test)

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, metrics_list)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, metrics_list)

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    model_name_lower = model_name.lower()
    if any(name in model_name_lower for name in ['ripper', 'rulefit']):
        understandability_result = calculate_ruleset_understandability(model, model_name)
    else:
        understandability_result = calculate_tree_understandability(model, model_name)

    return {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_confusion_matrix': train_cm,
        'test_confusion_matrix': test_cm,
        'classification_report': test_report,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'understandability': understandability_result
    }


def evaluate_all_models(trained_models, metrics_list=None):
    all_results = {}

    for model_name, model_data in trained_models.items():
        print(f"\nEvaluating {model_name}...")

        results = evaluate_model(
            model_data['model'],
            model_data['X_train'],
            model_data['X_test'],
            model_data['y_train'],
            model_data['y_test'],
            model_name,
            metrics_list
        )
        all_results[model_name] = results

    return all_results
