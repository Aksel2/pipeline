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
from sklearn.tree import export_text, plot_tree
import math
import re
from collections import defaultdict


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


def _predict_with_proba(model, X):
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError, ValueError):
            pass
    return y_pred, y_proba


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, metrics_list=None):
    y_train_pred, y_train_proba = _predict_with_proba(model, X_train)
    y_test_pred, y_test_proba = _predict_with_proba(model, X_test)

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, metrics_list)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, metrics_list)

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    model_name_lower = model_name.lower()
    if any(name in model_name_lower for name in ['ripper', 'rulefit', 'boosted_rules']):
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

        print(f"  Test Metrics: " + ", ".join(
            f"{m}={v:.4f}" for m, v in results['test_metrics'].items() if v is not None
        ))

    return all_results


def save_results_to_csv(all_results, output_dir):
    rows = []
    for model_name, results in all_results.items():
        row = {'model': model_name}

        for metric, value in results['train_metrics'].items():
            row[f'train_{metric}'] = value

        for metric, value in results['test_metrics'].items():
            row[f'test_{metric}'] = value

        if results.get('understandability'):
            u = results['understandability']
            row['understandability'] = u['understandability']
            row['understandability_x'] = u['x']
            row['understandability_N'] = u['N']
            row['understandability_D'] = u['D']
            row['understandability_DD'] = u['DD']
            row['understandability_F'] = u['F']

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"\nModel comparison saved to: {output_path}")

    return df


def plot_roc_curves(all_results, output_dir):
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


def _extract_tree_metrics(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature

    internal_count = 0
    leaf_depths = []
    feature_levels = defaultdict(list)

    def walk(node_id, depth):
        nonlocal internal_count
        is_leaf = children_left[node_id] == children_right[node_id]

        if is_leaf:
            leaf_depths.append(depth)
        else:
            internal_count += 1
            feature_levels[features[node_id]].append(depth)
            walk(children_left[node_id], depth + 1)
            walk(children_right[node_id], depth + 1)

    walk(0, 0)

    N = internal_count
    D = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0
    DD = sum(max(levels) - min(levels) for levels in feature_levels.values())
    F = len(feature_levels)

    return {"N": N, "D": D, "DD": DD, "F": F}



def _compute_understandability_score(N, D, DD, F, s=28, w1=1, w2=1, w3=1):
    x = w1 * (N + D) + w2 * DD + w3 * F
    understandability = math.exp(-((x / s) ** 2))
    return {
        "understandability": understandability,
        "N": N,
        "D": D,
        "DD": DD,
        "F": F,
        "x": x
    }


def calculate_tree_understandability(model, model_name, s=28, w1=1, w2=1, w3=1):
    model_name_lower = model_name.lower()

    all_metrics = []

    if 'figs' in model_name_lower:
        if hasattr(model, 'trees_'):
            for tree_obj in model.trees_:
                if hasattr(tree_obj, 'tree_') and hasattr(tree_obj.tree_, 'children_left'):
                    all_metrics.append(_extract_tree_metrics(tree_obj.tree_))
    elif hasattr(model, 'tree_') and hasattr(model.tree_, 'children_left'):
        all_metrics.append(_extract_tree_metrics(model.tree_))

    if not all_metrics:
        return None

    N = sum(m["N"] for m in all_metrics)
    D = sum(m["D"] for m in all_metrics) / len(all_metrics)
    DD = sum(m["DD"] for m in all_metrics)
    F = sum(m["F"] for m in all_metrics)

    # For sklearn trees with multiple trees (FIGS), recount unique features
    if 'figs' in model_name_lower and hasattr(model, 'trees_'):
        all_features = set()
        for tree_obj in model.trees_:
            if hasattr(tree_obj, 'tree_') and hasattr(tree_obj.tree_, 'children_left'):
                tree = tree_obj.tree_
                for i in range(tree.node_count):
                    if tree.children_left[i] != tree.children_right[i]:
                        all_features.add(tree.feature[i])
        F = len(all_features)
    elif hasattr(model, 'tree_') and hasattr(model.tree_, 'children_left'):
        F = all_metrics[0]["F"]

    return _compute_understandability_score(N, D, DD, F, s, w1, w2, w3)


def _extract_rules_as_feature_lists(model, model_name):
    model_name_lower = model_name.lower()

    if 'ripper' in model_name_lower:
        return _extract_ripper_feature_lists(model)
    elif 'rulefit' in model_name_lower:
        return _extract_rulefit_feature_lists(model)
    elif 'boosted_rules' in model_name_lower:
        return _extract_boosted_rules_feature_lists(model)

    return None


def _extract_ripper_feature_lists(model):
    if not hasattr(model, 'ruleset_') or not model.ruleset_:
        return None

    rules_features = []
    for rule in model.ruleset_.rules:
        if hasattr(rule, 'conds'):
            features = [cond.feature for cond in rule.conds]
            if features:
                rules_features.append(features)

    return rules_features if rules_features else None


def _extract_rulefit_feature_lists(model):
    try:
        rules_df = model._get_rules()
        rules_df = rules_df[rules_df['coef'] != 0]
        rules_df = rules_df[rules_df['type'] == 'rule']
    except Exception:
        return None

    if rules_df.empty:
        return None

    rules_features = []
    for _, row in rules_df.iterrows():
        rule_str = row['rule']
        features = re.findall(r'(\S+)\s*[<>=!]+', rule_str)
        if features:
            rules_features.append(features)

    return rules_features if rules_features else None


def _extract_boosted_rules_feature_lists(model):
    if not hasattr(model, 'estimators_'):
        return None

    rules_features = []
    feature_names = None
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_

    for estimator in model.estimators_:
        if hasattr(estimator, 'tree_'):
            tree = estimator.tree_
            features_in_rule = []
            for i in range(tree.node_count):
                if tree.children_left[i] != tree.children_right[i]:
                    feat_idx = tree.feature[i]
                    if feature_names is not None:
                        features_in_rule.append(feature_names[feat_idx])
                    else:
                        features_in_rule.append(str(feat_idx))
            if features_in_rule:
                rules_features.append(features_in_rule)

    return rules_features if rules_features else None


def _extract_ruleset_metrics(rules_as_feature_lists):
    if not rules_as_feature_lists:
        return None

    L = len(rules_as_feature_lists)
    N = sum(len(rule) for rule in rules_as_feature_lists)
    D = N / L if L > 0 else 0

    flattened = []
    for rule in rules_as_feature_lists:
        flattened.extend(rule)

    feature_positions = defaultdict(list)
    for i, feat in enumerate(flattened):
        feature_positions[feat].append(i)

    DD = 0
    for positions in feature_positions.values():
        for i in range(1, len(positions)):
            DD += positions[i] - positions[i - 1]

    F = len(feature_positions)

    return {"N": N, "D": D, "DD": DD, "F": F}


def calculate_ruleset_understandability(model, model_name, s=28, w1=1, w2=1, w3=1):
    rules_features = _extract_rules_as_feature_lists(model, model_name)
    if not rules_features:
        return None

    metrics = _extract_ruleset_metrics(rules_features)
    if not metrics:
        return None

    return _compute_understandability_score(
        metrics["N"], metrics["D"], metrics["DD"], metrics["F"], s, w1, w2, w3
    )


def extract_ripper_rules(model, feature_names=None):
    rules = ""
    if hasattr(model, 'ruleset_') and model.ruleset_:
        for i, rule in enumerate(model.ruleset_.rules, 1):
            rules += f"Rule {i}: {rule}\n"
    else:
        rules += "No rules learned (all instances classified as default class).\n"

    if hasattr(model, 'classes_'):
        default_class = model.classes_[0] if hasattr(model, 'class_order_') else "unknown"
        rules += f"\nDefault class: {default_class}"

    return rules



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


def extract_boosted_rules(model, feature_names=None):
    try:
        return str(model)
    except Exception as e:
        return f"Unable to extract rules from Boosted Rules model: {e}"


def extract_rulefit_rules(model, feature_names=None):
    try:
        rules = model._get_rules()
        rules = rules[rules['coef'] != 0].sort_values('importance', ascending=False)

        output = "Weighted Rules:\n"
        output += "=" * 60 + "\n"
        for _, row in rules.iterrows():
            output += f"IF {row['rule']}  weight={row['coef']:.4f}  importance={row['importance']:.4f}\n"

        return output if len(rules) > 0 else "No rules with non-zero coefficients."
    except Exception as e:
        return f"Unable to extract rules from RuleFit model: {e}"


def extract_model_rules(model, model_name, feature_names=None):
    model_name_lower = model_name.lower()

    if 'ripper' in model_name_lower:
        return extract_ripper_rules(model, feature_names)
    elif 'figs' in model_name_lower:
        return extract_figs_rules(model, feature_names)
    elif 'decision_tree' in model_name_lower or 'dtc' in model_name_lower:
        return extract_decision_tree_rules(model, feature_names)
    elif 'ebc' in model_name_lower or 'explainable' in model_name_lower:
        return extract_ebc_rules(model, feature_names)
    elif 'boosted_rules' in model_name_lower:
        return extract_boosted_rules(model, feature_names)
    elif 'rulefit' in model_name_lower:
        return extract_rulefit_rules(model, feature_names)
    else:
        return f"Rule extraction not implemented for model type: {model_name}"


def save_all_rules(trained_models, output_dir="outputs"):
    all_rules = {}

    combined_path = os.path.join(output_dir, "all_rules.txt")
    with open(combined_path, 'w') as combined_file:

        for model_name, model_data in trained_models.items():
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

    print(f"Rules saved to: {combined_path}")

    return all_rules


def plot_tree_visualizations(trained_models, output_dir):
    for model_name, model_data in trained_models.items():
        model = model_data['model']
        feature_names = list(model_data['X_train'].columns) if hasattr(model_data['X_train'], 'columns') else None
        model_name_lower = model_name.lower()

        if 'decision_tree' in model_name_lower or 'dtc' in model_name_lower:
            tree_depth = model.get_depth()
            fig_height = max(6, tree_depth * 3)
            fig_width = max(12, 2 ** tree_depth * 2)
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            plot_tree(model, feature_names=feature_names, class_names=['0', '1'],
                      filled=True, rounded=True, ax=ax, fontsize=8)
            ax.set_title(f'Decision Tree - {model_name}')
            output_path = os.path.join(output_dir, f'{model_name}_tree.png')
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Tree visualization saved to: {output_path}")



def run_evaluation(trained_models, config):
    evaluation_config = config.get("evaluation_config", {})
    metrics = evaluation_config.get("metrics", ["accuracy", "precision", "auroc"])
    output_dir = evaluation_config.get("output_directory", "outputs")

    os.makedirs(output_dir, exist_ok=True)

    all_results = evaluate_all_models(trained_models, metrics)

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

    try:
        plot_tree_visualizations(trained_models, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate tree visualizations: {e}")

    print(f"\n{comparison_df.to_string(index=False)}")

    return all_results, comparison_df
