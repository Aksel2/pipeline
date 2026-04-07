import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.tree import export_text


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

    if "auroc" in metrics_list or "roc_auc" in metrics_list:
        if y_pred_proba is not None:
            try:
                results["auroc"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                results["auroc"] = None
        else:
            results["auroc"] = None

    return results


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, metrics_list=None):
    y_train_pred = model.predict(X_train)
    y_train_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
        except (AttributeError, IndexError, ValueError):
            pass

    y_test_pred = model.predict(X_test)
    y_test_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError, ValueError):
            pass

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, metrics_list)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, metrics_list)

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    return {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_confusion_matrix': train_cm,
        'test_confusion_matrix': test_cm,
        'classification_report': test_report,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
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

        print(f"\nTrain Metrics:")
        for metric, value in results['train_metrics'].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")

        print(f"\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")

    return all_results


def save_results_to_csv(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for model_name, results in all_results.items():
        row = {'model': model_name}

        for metric, value in results['train_metrics'].items():
            row[f'train_{metric}'] = value

        for metric, value in results['test_metrics'].items():
            row[f'test_{metric}'] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"\nModel comparison saved to: {output_path}")

    return df


def plot_roc_curves(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))

    for model_name, results in all_results.items():
        y_test = results['y_test']
        y_test_proba = results['y_test_proba']

        if y_test_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            auc_score = results['test_metrics'].get('auroc', 0)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to: {output_path}")


def plot_confusion_matrices(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))

    if num_models == 1:
        axes = [axes]

    for idx, (model_name, results) in enumerate(all_results.items()):
        cm = results['test_confusion_matrix']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{model_name}\nConfusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to: {output_path}")


def save_classification_reports(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'classification_reports.txt')
    with open(output_path, 'w') as f:
        for model_name, results in all_results.items():
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'=' * 60}\n\n")

            report_dict = results['classification_report']

            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
            f.write(f"{'-' * 60}\n")

            for label in ['0', '1']:
                if label in report_dict:
                    metrics = report_dict[label]
                    f.write(f"{label:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                            f"{metrics['f1-score']:<12.4f} {metrics['support']:<12.0f}\n")

            f.write(f"\n")
            if 'accuracy' in report_dict:
                f.write(f"Accuracy: {report_dict['accuracy']:.4f}\n")

            if 'macro avg' in report_dict:
                macro = report_dict['macro avg']
                f.write(f"Macro avg - Precision: {macro['precision']:.4f}, "
                        f"Recall: {macro['recall']:.4f}, F1: {macro['f1-score']:.4f}\n")

            if 'weighted avg' in report_dict:
                weighted = report_dict['weighted avg']
                f.write(f"Weighted avg - Precision: {weighted['precision']:.4f}, "
                        f"Recall: {weighted['recall']:.4f}, F1: {weighted['f1-score']:.4f}\n")

    print(f"Classification reports saved to: {output_path}")


def extract_ripper_rules(model, feature_names=None):
    return str(model)


def extract_c45_rules(model, feature_names=None):
    try:
        return str(model)
    except:
        return "Unable to extract rules from C4.5 model"


def extract_figs_rules(model, feature_names=None):
    try:
        return str(model)
    except:
        return "Unable to extract rules from FIGS model"


def extract_decision_tree_rules(model, feature_names=None):
    if feature_names is not None:
        return export_text(model, feature_names=feature_names)
    return export_text(model)


def extract_ebc_rules(model, feature_names=None):
    try:
        importances = model.term_importances()
        feature_names_model = model.term_names_

        rules = "Feature Importances (Top 10):\n"
        rules += "=" * 50 + "\n"

        importance_pairs = list(zip(feature_names_model, importances))
        importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, importance) in enumerate(importance_pairs[:10], 1):
            rules += f"{i}. {feature}: {importance:.4f}\n"

        return rules
    except Exception as e:
        return f"Unable to extract interpretable components from EBC: {e}"


def extract_model_rules(model, model_name, feature_names=None):
    model_name_lower = model_name.lower()

    if 'ripper' in model_name_lower:
        return extract_ripper_rules(model, feature_names)
    elif 'c45' in model_name_lower or 'c4.5' in model_name_lower:
        return extract_c45_rules(model, feature_names)
    elif 'figs' in model_name_lower:
        return extract_figs_rules(model, feature_names)
    elif 'decision_tree' in model_name_lower or 'dtc' in model_name_lower:
        return extract_decision_tree_rules(model, feature_names)
    elif 'ebc' in model_name_lower or 'explainable' in model_name_lower:
        return extract_ebc_rules(model, feature_names)
    else:
        return f"Rule extraction not implemented for model type: {model_name}"


def save_all_rules(trained_models, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    all_rules = {}

    print("\n" + "=" * 60)
    print("Extracting Rules from Models")
    print("=" * 60)

    combined_path = os.path.join(output_dir, "all_rules.txt")
    with open(combined_path, 'w') as combined_file:

        for model_name, model_data in trained_models.items():
            print(f"\nExtracting rules from {model_name}...")

            model = model_data['model']
            X_train = model_data['X_train']

            feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None

            rules = extract_model_rules(model, model_name, feature_names)
            all_rules[model_name] = rules

            combined_file.write("=" * 60 + "\n")
            combined_file.write(f"Model: {model_name}\n")
            combined_file.write("=" * 60 + "\n\n")
            combined_file.write(rules)
            combined_file.write("\n\n")

            individual_path = os.path.join(output_dir, f"{model_name}_rules.txt")
            with open(individual_path, 'w') as f:
                f.write(f"Rules for {model_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(rules)

            print(f"  Saved to: {individual_path}")

            print(f"\n  Preview:")
            preview_lines = rules.split('\n')[:10]
            for line in preview_lines:
                print(f"    {line}")
            if len(rules.split('\n')) > 10:
                print(f"    ... (see full rules in {individual_path})")

    print(f"\nAll rules saved to: {combined_path}")
    print("=" * 60)

    return all_rules


def run_evaluation(trained_models, config):
    evaluation_config = config.get("evaluation_config", {})
    metrics = evaluation_config.get("metrics", ["accuracy", "precision", "auroc"])
    output_dir = evaluation_config.get("output_directory", "outputs")

    print("\n" + "=" * 60)
    print("Starting Model Evaluation")
    print("=" * 60)

    all_results = evaluate_all_models(trained_models, metrics)

    print("\n" + "=" * 60)
    print("Saving Evaluation Results")
    print("=" * 60)

    comparison_df = save_results_to_csv(all_results, output_dir)
    save_classification_reports(all_results, output_dir)

    try:
        plot_roc_curves(all_results, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate ROC curves: {e}")

    try:
        plot_confusion_matrices(all_results, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate confusion matrices: {e}")

    try:
        rules_dict = save_all_rules(trained_models, output_dir)
    except Exception as e:
        print(f"Warning: Could not extract rules: {e}")

    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    return all_results, comparison_df
