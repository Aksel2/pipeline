import json
import os
from pathlib import Path

import pandas as pd

from preprocessing.bpm_graph import BPMNGraph


VALID_BALANCING_STRATEGIES = {"none", "undersample", "oversample"}
VALID_METRICS = {"accuracy", "precision", "recall", "f1", "auroc"}
VALID_MODELS = {
    "decision_tree_classifier",
    "figs_classifier",
    "ripper_classifier",
    "rulefit_classifier",
    "explainable_boosting_classifier",
}


def check_config(config):
    preprocess_config = config.get("preprocess_config")
    if not preprocess_config:
        raise ValueError("Configuration must contain 'preprocess_config'")

    if preprocess_config.get("bpmn_model_path"):
        if not preprocess_config.get("target_gateway_id"):
            raise ValueError(
                "'target_gateway_id' is required when 'bpmn_model_path' is set"
            )
        outcome_mapping = preprocess_config.get("outcome_mapping")
        if not outcome_mapping:
            raise ValueError(
                "'outcome_mapping' is required when 'bpmn_model_path' is set"
            )
        for flow_id, outcome in outcome_mapping.items():
            if outcome not in (0, 1):
                raise ValueError(
                    f"'outcome_mapping' values must be 0 or 1, got {outcome!r} for flow {flow_id!r}"
                )
    else:
        for field in (
            "pre_decision_activities",
            "post_decision_0_activities",
            "post_decision_1_activities",
        ):
            if not preprocess_config.get(field):
                raise ValueError(
                    f"'{field}' must be a non-empty list when not in BPMN mode"
                )

    strategy = config.get("balancing_config", {}).get("strategy", "none")
    if strategy not in VALID_BALANCING_STRATEGIES:
        raise ValueError(
            f"'balancing_config.strategy' must be one of "
            f"{sorted(VALID_BALANCING_STRATEGIES)}, got {strategy!r}"
        )

    for model_name in config.get("models_config", {}):
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model {model_name!r} in 'models_config'. "
                f"Valid models: {sorted(VALID_MODELS)}"
            )

    for metric in config.get("evaluation_config", {}).get("metrics", []):
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r} in 'evaluation_config.metrics'. "
                f"Valid metrics: {sorted(VALID_METRICS)}"
            )

    encoding_config = config.get("encoding_config", {})
    aggregation = [c.lower() for c in encoding_config.get("encoding_strategies", {}).get("aggregation", [])]
    continuous = [c.lower() for c in encoding_config.get("event_attributes_continuous", [])]
    invalid = [c for c in aggregation if c not in continuous]
    if invalid:
        raise ValueError(
            f"Columns in 'encoding_strategies.aggregation' must be declared in "
            f"'event_attributes_continuous': {invalid}"
        )


def check_event_log(df, config, source):
    columns_lower = {c.lower() for c in df.columns}
    preprocess_config = config["preprocess_config"]
    column_names = preprocess_config.get("column_names", {})

    required = {
        "case_id": column_names.get("case_id", "case_id"),
        "activity": column_names.get("activity", "activity"),
        "start_time": column_names.get("start_time", "start_time"),
        "end_time": column_names.get("end_time", "end_time"),
    }
    missing = [
        f"'{col}' (configured as {label})"
        for label, col in required.items()
        if col.lower() not in columns_lower
    ]
    if missing:
        raise ValueError(f"Required columns not found in {source}: {', '.join(missing)}")

    encoding_config = config.get("encoding_config", {})
    for field in ("case_attributes", "event_attributes_continuous", "event_attributes_discrete"):
        for col in encoding_config.get(field, []):
            if col.lower() not in columns_lower:
                raise ValueError(
                    f"Column '{col}' from 'encoding_config.{field}' not found in {source}"
                )

    if not preprocess_config.get("bpmn_model_path"):
        activity_col_name = column_names.get("activity", "activity").lower()
        matching = [c for c in df.columns if c.lower() == activity_col_name]
        if matching:
            activities = set(df[matching[0]].dropna().unique())
            for field in (
                "pre_decision_activities",
                "post_decision_0_activities",
                "post_decision_1_activities",
            ):
                absent = [a for a in preprocess_config.get(field, []) if a not in activities]
                if absent:
                    raise ValueError(
                        f"Activities in 'preprocess_config.{field}' not found in {source}: {absent}"
                    )


def check_bpmn(bpmn_graph, config):
    preprocess_config = config["preprocess_config"]
    target_gateway_id = preprocess_config["target_gateway_id"]
    outcome_mapping = preprocess_config["outcome_mapping"]

    if target_gateway_id not in bpmn_graph.element_info:
        raise ValueError(
            f"target_gateway_id '{target_gateway_id}' not found in BPMN model"
        )

    target = bpmn_graph.element_info[target_gateway_id]
    if not target.is_gateway():
        raise ValueError(
            f"target_gateway_id '{target_gateway_id}' is not a gateway "
            f"(it is of type {target.type.value})"
        )

    outgoing = set(target.outgoing_flows)
    mapped = set(outcome_mapping.keys())

    unknown = mapped - outgoing
    if unknown:
        raise ValueError(
            f"'outcome_mapping' contains flow IDs that are not outgoing flows of "
            f"target gateway '{target_gateway_id}': {sorted(unknown)}"
        )

    unmapped = outgoing - mapped
    if unmapped:
        raise ValueError(
            f"'outcome_mapping' must cover all outgoing flows of target gateway "
            f"'{target_gateway_id}'. Missing: {sorted(unmapped)}"
        )


def validate_inputs(input_logs_path, test_logs_path, config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(input_logs_path):
        raise FileNotFoundError(f"Input log file not found: {input_logs_path}")
    if test_logs_path is not None and not os.path.isfile(test_logs_path):
        raise FileNotFoundError(f"Test log file not found: {test_logs_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    check_config(config)

    bpmn_path = config["preprocess_config"].get("bpmn_model_path", "")
    if bpmn_path and not os.path.isfile(bpmn_path):
        raise FileNotFoundError(f"BPMN model file not found: {bpmn_path}")

    train_data = pd.read_csv(input_logs_path)
    check_event_log(train_data, config, source=input_logs_path)

    test_data = None
    if test_logs_path is not None:
        test_data = pd.read_csv(test_logs_path)
        check_event_log(test_data, config, source=test_logs_path)

    if bpmn_path:
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))
        check_bpmn(bpmn_graph, config)

    return config, train_data, test_data
