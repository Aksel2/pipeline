import pandas as pd

from decision_pipeline.model_evaluate.metrics import evaluate_all_models
from decision_pipeline.model_evaluate.plots import (
    plot_roc_curves,
    plot_tree_visualizations,
)
from decision_pipeline.model_evaluate.reports import (
    save_all_rules,
    save_classification_reports,
    save_results_to_csv,
)


def format_understandability(x):
    return f"{x:.3g}" if pd.notna(x) else ""


def _resolve_class_names(config):
    pre = config.get("preprocess_config", {})
    cls0 = pre.get("post_decision_0_activities") or []
    cls1 = pre.get("post_decision_1_activities") or []
    if cls0 and cls1:
        return [cls0[0], cls1[0]]
    return None


def _resolve_understandability_kwargs(understandability_cfg):
    allowed = {"s", "w1", "w2", "w3"}
    shared = {k: v for k, v in understandability_cfg.items() if k in allowed}
    tree_override = {k: v for k, v in understandability_cfg.get("tree", {}).items() if k in allowed}
    ruleset_override = {k: v for k, v in understandability_cfg.get("ruleset", {}).items() if k in allowed}
    tree_kwargs = {**shared, **tree_override}
    ruleset_kwargs = {**shared, **ruleset_override}
    return tree_kwargs, ruleset_kwargs


def run_evaluation(trained_models, config):
    evaluation_config = config.get("evaluation_config", {})
    metrics = evaluation_config.get("metrics", ["accuracy", "auroc"])
    output_dir = evaluation_config.get("output_directory", "outputs")

    understandability_cfg = evaluation_config.get("understandability_config", {})
    tree_kwargs, ruleset_kwargs = _resolve_understandability_kwargs(understandability_cfg)

    all_results = evaluate_all_models(trained_models, metrics, tree_kwargs, ruleset_kwargs)

    comparison_df = save_results_to_csv(all_results, output_dir)
    save_classification_reports(all_results, output_dir)

    plot_roc_curves(all_results, output_dir)
    save_all_rules(trained_models, output_dir)
    plot_tree_visualizations(trained_models, output_dir, _resolve_class_names(config))

    formatters = {}
    if 'understandability' in comparison_df.columns:
        formatters['understandability'] = format_understandability
    print(f"\n{comparison_df.to_string(index=False, formatters=formatters)}")
    print(f"\nAll outputs saved to: {output_dir}")

    return all_results, comparison_df
