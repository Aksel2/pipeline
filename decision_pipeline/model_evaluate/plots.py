import os

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree


def _simplify_tree_node_text(ax):
    for child in ax.get_children():
        if not isinstance(child, Annotation):
            continue
        text = child.get_text()
        if not text:
            continue
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            continue
        if '<=' in lines[0]:
            child.set_text(lines[0])
        else:
            child.set_text(lines[-1])


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


def _layout_figs_tree(node, depth, x_counter):
    if node is None:
        return None
    is_leaf = node.left is None and node.right is None
    if is_leaf:
        x = x_counter[0]
        x_counter[0] += 1
        return {'node': node, 'x': x, 'y': -depth, 'is_leaf': True, 'children': []}
    left = _layout_figs_tree(node.left, depth + 1, x_counter)
    right = _layout_figs_tree(node.right, depth + 1, x_counter)
    x = (left['x'] + right['x']) / 2.0
    return {'node': node, 'x': x, 'y': -depth, 'is_leaf': False, 'children': [left, right]}


def _figs_tree_depth(node):
    if node is None or (node.left is None and node.right is None):
        return 0
    return 1 + max(_figs_tree_depth(node.left), _figs_tree_depth(node.right))


def _figs_tree_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return _figs_tree_leaves(node.left) + _figs_tree_leaves(node.right)


def _draw_figs_layout(layout, ax, feature_names):
    def _format_value(val):
        try:
            arr = list(val)
            if len(arr) == 1:
                return f"{float(arr[0]):.2f}"
            return "[" + ", ".join(f"{float(v):.2f}" for v in arr) + "]"
        except TypeError:
            return f"{float(val):.2f}"

    def walk(layout_node):
        node = layout_node['node']
        x, y = layout_node['x'], layout_node['y']
        if layout_node['is_leaf']:
            label = _format_value(node.value)
            ax.annotate(
                label, (x, y), ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', fc='lightblue', ec='navy', lw=0.8),
            )
        else:
            fname = feature_names[node.feature] if feature_names and node.feature is not None else f"X_{node.feature}"
            label = f"{fname} <= {node.threshold:.2f}"
            ax.annotate(
                label, (x, y), ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', fc='wheat', ec='black', lw=0.8),
            )
            for child in layout_node['children']:
                ax.plot([x, child['x']], [y - 0.15, child['y'] + 0.15], 'k-', lw=0.6)
                walk(child)

    walk(layout)


def plot_figs_trees(model, feature_names, output_path):
    trees = getattr(model, 'trees_', None)
    if not trees:
        return

    n_trees = len(trees)
    leaf_counts = [max(1, _figs_tree_leaves(t)) for t in trees]
    depths = [max(1, _figs_tree_depth(t)) for t in trees]
    widths = [max(3, min(20, lc * 2.0)) for lc in leaf_counts]
    height = max(4, min(16, max(depths) * 2.0))

    fig, axes = plt.subplots(1, n_trees, figsize=(sum(widths), height), squeeze=False)
    axes = axes[0]

    for ax, tree, idx in zip(axes, trees, range(n_trees)):
        layout = _layout_figs_tree(tree, 0, [0])
        if layout is not None:
            _draw_figs_layout(layout, ax, feature_names)
            ax.set_title(f'Tree #{idx}')
        ax.set_axis_off()
        ax.margins(0.15, 0.25)

    fig.suptitle(f'FIGS ({n_trees} tree{"s" if n_trees != 1 else ""})')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tree_visualizations(trained_models, output_dir, class_names=None):
    for model_name, model_data in trained_models.items():
        model = model_data['model']
        feature_names = list(model_data['X_train'].columns) if hasattr(model_data['X_train'], 'columns') else None

        if model_name == 'decision_tree':
            tree_depth = model.get_depth()
            n_leaves = model.get_n_leaves() if hasattr(model, 'get_n_leaves') else 2 ** tree_depth
            fig_width = max(8, min(40, n_leaves * 1.8))
            fig_height = max(6, min(20, tree_depth * 1.8))
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            plot_tree(
                model,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                ax=ax,
                fontsize=8,
                impurity=False,
                label='none',
            )
            _simplify_tree_node_text(ax)
            ax.set_title('Decision Tree')
            output_path = os.path.join(output_dir, f'{model_name}_tree.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        elif model_name == 'figs':
            output_path = os.path.join(output_dir, f'{model_name}_tree.png')
            plot_figs_trees(model, feature_names, output_path)
