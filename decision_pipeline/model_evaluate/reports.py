import os

import pandas as pd
from sklearn.metrics import classification_report

from decision_pipeline.model_evaluate.rule_extraction import extract_model_rules


def save_results_to_csv(all_results, output_dir):
    rows = []
    for model_name, results in all_results.items():
        row = {'model': model_name}

        for metric, value in results['test_metrics'].items():
            row[f'test_{metric}'] = value

        if results.get('understandability'):
            row['understandability'] = results['understandability']['understandability']

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(output_path, index=False)

    return df


def save_classification_reports(all_results, output_dir):
    output_path = os.path.join(output_dir, 'classification_reports.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for model_name, results in all_results.items():
            f.write(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}\n\n")
            f.write(classification_report(results['y_test'], results['y_test_pred'], zero_division=0))
            f.write("\n")


def save_all_rules(trained_models, output_dir="outputs"):
    combined_path = os.path.join(output_dir, "all_rules.txt")
    with open(combined_path, 'w', encoding='utf-8') as combined_file:

        for model_name, model_data in trained_models.items():
            model = model_data['model']

            rules = extract_model_rules(model, model_name)

            combined_file.write("=" * 60 + "\n")
            combined_file.write(f"Model: {model_name}\n")
            combined_file.write("=" * 60 + "\n\n")
            combined_file.write(rules)
            combined_file.write("\n\n")

            individual_path = os.path.join(output_dir, f"{model_name}_rules.txt")
            with open(individual_path, 'w', encoding='utf-8') as f:
                f.write(f"Rules for {model_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(rules)
