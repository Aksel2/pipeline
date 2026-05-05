import pandas as pd
from sklearn.model_selection import train_test_split

from encoding.encoding import collapse_observations, DataFrameEncoder
from model_evaluate.evaluate import run_evaluation
from model_train.models import prepare_features_and_target, train_dtc, train_ripper, train_figs, train_ebc, train_rulefit
from preprocessing.preprocess import preprocess_event_log_replay, preprocess_event_log
from validation import validate_inputs

MODEL_REGISTRY = [
    ("decision_tree_classifier", "Decision Tree Classifier", train_dtc, "decision_tree"),
    ("ripper_classifier", "RIPPER Classifier", train_ripper, "ripper"),
    ("figs_classifier", "FIGS Classifier", train_figs, "figs"),
    ("rulefit_classifier", "RuleFit Classifier", train_rulefit, "rulefit"),
    ("explainable_boosting_classifier", "Explainable Boosting Classifier", train_ebc, "ebc"),
]


def balance_dataset(df, strategy):
    valid_strategies = {"none", "undersample", "oversample"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid balancing strategy '{strategy}'. Must be one of: {sorted(valid_strategies)}"
        )

    if strategy == "none":
        return df

    class_counts = df['target'].value_counts()
    if len(class_counts) != 2 or class_counts.min() == class_counts.max():
        return df

    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_df = df[df['target'] == minority_class]
    majority_df = df[df['target'] == majority_class]

    if strategy == "undersample":
        majority_df = majority_df.sample(n=len(minority_df), random_state=42)
        balanced = pd.concat([minority_df, majority_df], ignore_index=True)
        print(f"\nUndersampled class {majority_class} from {class_counts.max()} to {len(minority_df)}")
    else:
        minority_df = minority_df.sample(n=len(majority_df), replace=True, random_state=42)
        balanced = pd.concat([minority_df, majority_df], ignore_index=True)
        print(f"\nOversampled class {minority_class} from {class_counts.min()} to {len(majority_df)}")

    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced


def preprocess_and_encode(df_train, df_test, config, balancing_strategy):
    df_train = collapse_observations(df_train, config)
    df_test = collapse_observations(df_test, config)

    df_train['target'] = pd.to_numeric(df_train['target'])
    df_test['target'] = pd.to_numeric(df_test['target'])

    df_train = balance_dataset(df_train, balancing_strategy)

    encoder = DataFrameEncoder(config)
    df_train_final = encoder.fit_transform(df_train)
    df_test_final = encoder.transform(df_test)

    return df_train_final, df_test_final


def pipeline(
        input_logs_path,
        test_logs_path=None,
        test_percentage=None,
        config_path="config.json"
):
    config, data, test_data = validate_inputs(input_logs_path, test_logs_path, config_path)

    preprocess_config = config.get("preprocess_config", {})
    use_replay = bool(preprocess_config.get("bpmn_model_path"))
    if use_replay:
        print("Using BPMN replay-based preprocessing")
        preprocess_fn = preprocess_event_log_replay
    else:
        preprocess_fn = preprocess_event_log

    balancing_strategy = config.get("balancing_config", {}).get("strategy", "none")

    if test_logs_path is not None:
        print(f"Using separate test file: {test_logs_path}")
        df_train = preprocess_fn(data, config)
        df_test = preprocess_fn(test_data, config)
    else:
        if test_percentage is None:
            test_percentage = 0.2
        print(f"Splitting data with test_percentage: {test_percentage}")
        df = preprocess_fn(data, config)
        df.to_csv('preprocessed_data.csv', index=False)
        df_filtered = collapse_observations(df, config)
        df_filtered['target'] = pd.to_numeric(df_filtered['target'])
        df_train, df_test = train_test_split(
            df_filtered,
            test_size=test_percentage,
            random_state=42,
            stratify=df_filtered['target']
        )

    if test_logs_path is not None:
        df_train_final, df_test_final = preprocess_and_encode(df_train, df_test, config, balancing_strategy)
    else:
        df_train = balance_dataset(df_train, balancing_strategy)
        encoder = DataFrameEncoder(config)
        df_train_final = encoder.fit_transform(df_train)
        df_test_final = encoder.transform(df_test)

    df_train_final.to_csv('train_encoded.csv', index=False)
    df_test_final.to_csv('test_encoded.csv', index=False)

    print(f"\nTrain set: {len(df_train_final)}")
    print(f"Test set: {len(df_test_final)}")

    X_train, y_train = prepare_features_and_target(df_train_final, config)
    X_test, y_test = prepare_features_and_target(df_test_final, config)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    models_config = config.get("models_config", {})
    trained_models = {}

    for config_key, display_name, train_fn, model_key in MODEL_REGISTRY:
        if not models_config.get(config_key, {}).get("enabled", False):
            continue

        print(f"\nTraining {display_name}...")

        model_params = models_config.get(config_key, {})
        model = train_fn(X_train, X_test, y_train, y_test, model_params)
        trained_models[model_key] = {
            'model': model,
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
        input_logs_path="new_data/renewed_logs.csv",
        test_logs_path=None,
        test_percentage=0.2,
        config_path="config.json"
    )

    print(f"\nPipeline complete. Trained {len(trained_models)} models.")
