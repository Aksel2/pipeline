import os

import pandas as pd

from model_evaluate.rule_extraction import extract_model_rules


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

    return df


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


def save_all_rules(trained_models, output_dir="outputs"):
    combined_path = os.path.join(output_dir, "all_rules.txt")
    with open(combined_path, 'w') as combined_file:

        for model_name, model_data in trained_models.items():
            model = model_data['model']
            X_train = model_data['X_train']

            feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None

            rules = extract_model_rules(model, model_name, feature_names)

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
