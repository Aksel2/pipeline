import json

import pandas as pd
from sklearn.model_selection import train_test_split

from encoding.encoding import collapse_observations, DataFrameEncoder
from model_evaluate.evaluate import run_evaluation
from model_train.models import prepare_features_and_target, train_dtc, train_ripper, train_c45, train_figs, train_ebc
from preprocessing.preprocess import preprocess_event_log_replay, preprocess_event_log


def pipeline(
        input_logs_path: str,
        test_logs_path: str = None,
        test_percentage: float = None,
        config_path: str = "config.json"
):
    with open(config_path, 'r') as f:
        config = json.load(f)

    data = pd.read_csv(input_logs_path)

    preprocess_config = config.get("preprocess_config", {})

    use_replay = bool(preprocess_config.get("bpmn_model_path"))
    if use_replay:
        print("Using BPMN replay-based preprocessing")
        preprocess_fn = preprocess_event_log_replay
    else:
        preprocess_fn = preprocess_event_log

    pre_decision_activities = preprocess_config.get("pre_decision_activities", [])
    if not use_replay and not pre_decision_activities:
        raise ValueError("Configuration must contain at least one pre_decision_activity")

    if test_logs_path is not None:
        print(f"Using separate test file: {test_logs_path}")
        test_data = pd.read_csv(test_logs_path)

        df_train = preprocess_fn(data, config)
        df_test = preprocess_fn(test_data, config)

        df_train_filtered = collapse_observations(df_train, config)
        df_test_filtered = collapse_observations(df_test, config)

        df_train_filtered['target'] = pd.to_numeric(df_train_filtered['target'])
        df_test_filtered['target'] = pd.to_numeric(df_test_filtered['target'])

        encoder = DataFrameEncoder(config)
        df_train_final = encoder.fit_transform(df_train_filtered)
        df_test_final = encoder.transform(df_test_filtered)

    else:
        if test_percentage is None:
            test_percentage = 0.2

        print(f"Splitting data with test_percentage: {test_percentage}")

        df = preprocess_fn(data, config)
        df.to_csv('preprocessed_data.csv', index=False)

        df_filtered = collapse_observations(df, config)

        df_filtered['target'] = pd.to_numeric(df_filtered['target'])

        df_train_split, df_test_split = train_test_split(
            df_filtered,
            test_size=test_percentage,
            random_state=42,
            stratify=df_filtered['target']
        )

        encoder = DataFrameEncoder(config)
        df_train_final = encoder.fit_transform(df_train_split)
        df_test_final = encoder.transform(df_test_split)

    df_train_final.to_csv('train_encoded.csv', index=False)
    df_test_final.to_csv('test_encoded.csv', index=False)

    print(f"\nTrain set size: {len(df_train_final)}")
    print(f"Test set size: {len(df_test_final)}")
    print(f"Train target distribution:\n{df_train_final['target'].value_counts()}")
    print(f"Test target distribution:\n{df_test_final['target'].value_counts()}")

    X_train, y_train = prepare_features_and_target(df_train_final, config)
    X_test, y_test = prepare_features_and_target(df_test_final, config)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    models_config = config.get("models_config", {})
    trained_models = {}

    if models_config.get("decision_tree_classifier", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("Training Decision Tree Classifier")
        print("=" * 50)
        model_params = models_config.get("decision_tree_classifier", {})
        dtc_model = train_dtc(X_train, X_test, y_train, y_test, model_params)
        trained_models['decision_tree'] = {
            'model': dtc_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    if models_config.get("ripper_classifier", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("Training RIPPER Classifier")
        print("=" * 50)
        model_params = models_config.get("ripper_classifier", {})
        ripper_model = train_ripper(X_train, X_test, y_train, y_test, model_params)
        trained_models['ripper'] = {
            'model': ripper_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    if models_config.get("c45_tree_classifier", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("Training C4.5 Tree Classifier")
        print("=" * 50)
        model_params = models_config.get("c45_tree_classifier", {})
        c45_model = train_c45(X_train, X_test, y_train, y_test, model_params)
        trained_models['c45'] = {
            'model': c45_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    if models_config.get("figs_classifier", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("Training FIGS Classifier")
        print("=" * 50)
        model_params = models_config.get("figs_classifier", {})
        figs_model = train_figs(X_train, X_test, y_train, y_test, model_params)
        trained_models['figs'] = {
            'model': figs_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    if models_config.get("explainable_boosting_classifier", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("Training Explainable Boosting Classifier")
        print("=" * 50)
        model_params = models_config.get("explainable_boosting_classifier", {})
        ebc_model = train_ebc(X_train, X_test, y_train, y_test, model_params)
        trained_models['ebc'] = {
            'model': ebc_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    if trained_models:
        evaluation_results, comparison_df = run_evaluation(trained_models, config)
    else:
        print("\nWarning: No models were trained. Skipping evaluation.")
        evaluation_results = {}
        comparison_df = None

    return df_train_final, df_test_final, trained_models, evaluation_results


if __name__ == "__main__":
    train_df, test_df, trained_models, evaluation_results = pipeline(
        input_logs_path="data/data.csv",
        test_logs_path=None,
        test_percentage=0.2,
        config_path="config.json"
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Trained {len(trained_models)} models:")
    for model_name in trained_models.keys():
        print(f"  - {model_name}")
    print(f"\nEvaluation results saved to outputs/")
    print("=" * 60)
