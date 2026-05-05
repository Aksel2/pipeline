import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree


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
