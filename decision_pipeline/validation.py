import json
import os
from pathlib import Path

import jsonschema
import pandas as pd

from decision_pipeline.preprocessing.bpm_graph import BPMNGraph


_SCHEMA_PATH = Path(__file__).resolve().parent / "config.schema.json"


def _load_schema():
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_config(config):
    schema = _load_schema()
    try:
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValueError(f"Config validation failed at '{path}': {e.message}")

    preprocess_config = config["preprocess_config"]
    mode = preprocess_config["type"]

    if mode == "bpmn":
        for field in ("bpmn_model_path", "target_gateway_id", "outcome_mapping"):
            if not preprocess_config.get(field):
                raise ValueError(f"'{field}' is required when 'type' is 'bpmn'")
    else:
        for field in (
            "pre_decision_activities",
            "post_decision_0_activities",
            "post_decision_1_activities",
        ):
            if not preprocess_config.get(field):
                raise ValueError(
                    f"'{field}' must be a non-empty list when 'type' is 'log'"
                )

    understandability_cfg = config.get("evaluation_config", {}).get("understandability_config")
    if understandability_cfg is not None:
        weights_present = [k for k in ("w1", "w2", "w3") if k in understandability_cfg]
        if 0 < len(weights_present) < 3:
            missing = sorted({"w1", "w2", "w3"} - set(weights_present))
            raise ValueError(
                f"'understandability_config' weights w1, w2, w3 must be set together or not at all. "
                f"Missing: {missing}"
            )
        if len(weights_present) == 3:
            weights_sum = sum(understandability_cfg[k] for k in weights_present)
            if abs(weights_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"'understandability_config' weights w1+w2+w3 must sum to 1, got {weights_sum}"
                )

    encoding_config = config.get("encoding_config", {})
    strategies = encoding_config.get("encoding_strategies", {})
    continuous = {c.lower() for c in encoding_config.get("event_attributes_continuous", [])}
    categorical = {c.lower() for c in encoding_config.get("case_attributes", [])} | {
        c.lower() for c in encoding_config.get("event_attributes_discrete", [])
    }

    invalid_agg = [c for c in strategies.get("aggregation", []) if c.lower() not in continuous]
    if invalid_agg:
        raise ValueError(
            f"Columns in 'encoding_strategies.aggregation' must be declared in "
            f"'event_attributes_continuous': {invalid_agg}"
        )
    for key in ("one-hot", "target-based"):
        invalid = [c for c in strategies.get(key, []) if c.lower() not in categorical]
        if invalid:
            raise ValueError(
                f"Columns in 'encoding_strategies.{key}' must be declared in "
                f"'case_attributes' or 'event_attributes_discrete': {invalid}"
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

    if preprocess_config.get("type") == "log":
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

    with open(config_path, "r") as f:
        config = json.load(f)
    check_config(config)

    preprocess_config = config["preprocess_config"]
    if preprocess_config["type"] == "bpmn":
        bpmn_path = preprocess_config["bpmn_model_path"]
        if not os.path.isfile(bpmn_path):
            raise FileNotFoundError(f"BPMN model file not found: {bpmn_path}")

    train_data = pd.read_csv(input_logs_path)
    check_event_log(train_data, config, source=input_logs_path)

    test_data = None
    if test_logs_path is not None:
        test_data = pd.read_csv(test_logs_path)
        check_event_log(test_data, config, source=test_logs_path)

    if preprocess_config["type"] == "bpmn":
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(preprocess_config["bpmn_model_path"]))
        check_bpmn(bpmn_graph, config)

    return config, train_data, test_data
