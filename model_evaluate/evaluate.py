import os

import pandas as pd

from model_evaluate.metrics import evaluate_all_models
from model_evaluate.plots import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_tree_visualizations,
)
from model_evaluate.reports import (
    save_all_rules,
    save_classification_reports,
    save_results_to_csv,
)


def format_understandability(x):
    return f"{x:.3g}" if pd.notna(x) else ""


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
        save_all_rules(trained_models, output_dir)
    except Exception as e:
        print(f"Warning: Could not extract rules: {e}")

    try:
        plot_tree_visualizations(trained_models, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate tree visualizations: {e}")

    formatters = {}
    if 'understandability' in comparison_df.columns:
        formatters['understandability'] = format_understandability
    print(f"\n{comparison_df.to_string(index=False, formatters=formatters)}")
    print(f"\nAll outputs saved to: {output_dir}")

    return all_results, comparison_df
