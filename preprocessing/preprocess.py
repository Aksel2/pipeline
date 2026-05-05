from pathlib import Path

import pandas as pd

from preprocessing.bpm_graph import BPMNNodeType, BPMNGraph


def extract_column_mapping(config):
    preprocess_config = config.get("preprocess_config", {})
    column_mapping = preprocess_config.get("column_names", {})
    return (
        column_mapping.get("case_id", "case_id").lower(),
        column_mapping.get("activity", "activity").lower(),
        column_mapping.get("start_time", "start_time").lower(),
        column_mapping.get("end_time", "end_time").lower(),
    )


def extract_attributes(config):
    encoding_config = config.get("encoding_config", {})
    case_attributes = [attr.lower() for attr in encoding_config.get("case_attributes", [])]
    event_attributes_continuous = [attr.lower() for attr in encoding_config.get("event_attributes_continuous", [])]
    event_attributes_discrete = [attr.lower() for attr in encoding_config.get("event_attributes_discrete", [])]
    return case_attributes, event_attributes_continuous, event_attributes_discrete


def prepare_event_log(df, config):
    df = df.copy()
    df.columns = df.columns.str.lower()

    case_id_col, activity_col, start_time_col, end_time_col = extract_column_mapping(config)

    case_attributes, event_attributes_continuous, event_attributes_discrete = extract_attributes(config)

    columns_to_keep = [case_id_col, activity_col, start_time_col, end_time_col]
    for attr in case_attributes + event_attributes_continuous + event_attributes_discrete:
        if attr in df.columns and attr not in columns_to_keep:
            columns_to_keep.append(attr)

    df_sorted = df.sort_values([case_id_col, start_time_col])
    df_sorted = df_sorted[columns_to_keep]

    return df_sorted, case_id_col, activity_col


def empty_result(df_sorted):
    empty = df_sorted.iloc[0:0].copy()
    empty['observation'] = pd.array([], dtype=pd.Int64Dtype())
    empty['target'] = pd.array([], dtype=pd.Int64Dtype())
    return empty


def preprocess_event_log_replay(df, config):
    df_sorted, case_id_col, activity_col = prepare_event_log(df, config)

    preprocess_config = config.get("preprocess_config", {})
    bpmn_model_path = preprocess_config["bpmn_model_path"]
    target_gateway_id = preprocess_config["target_gateway_id"]
    outcome_mapping = {k: int(v) for k, v in preprocess_config["outcome_mapping"].items()}

    bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_model_path))

    f_arcs_frequency = {}

    bpmn_task_names = set()
    for e_id, e_info in bpmn_graph.element_info.items():
        if e_info.type == BPMNNodeType.TASK:
            bpmn_task_names.add(e_info.name)

    observation_rows = []

    for case_id, group in df_sorted.groupby(case_id_col):
        indices = group.index.tolist()
        all_activities = group[activity_col].tolist()

        task_indices = []
        task_sequence = []
        for i, act in enumerate(all_activities):
            if act in bpmn_task_names:
                task_indices.append(i)
                task_sequence.append(act)

        if not task_sequence:
            continue

        bpmn_graph.gateway_states = {}

        is_correct, fired_tasks, pending, gateway_decisions = bpmn_graph.replay_trace(
            task_sequence, f_arcs_frequency,
            target_gateway_id=target_gateway_id,
            outcome_mapping=outcome_mapping
        )

        if not gateway_decisions:
            continue

        observation_count = 0
        for task_index, outcome in gateway_decisions:
            observation_count += 1
            original_pos = task_indices[task_index]
            obs_rows = df_sorted.iloc[[indices[i] for i in range(original_pos + 1)]].copy()
            obs_rows['observation'] = observation_count
            obs_rows['target'] = outcome
            observation_rows.append(obs_rows)

    if not observation_rows:
        return empty_result(df_sorted)

    return pd.concat(observation_rows, ignore_index=True)


def preprocess_event_log(df, config):
    df_sorted, case_id_col, activity_col = prepare_event_log(df, config)

    preprocess_config = config.get("preprocess_config", {})

    pre_decision_list = preprocess_config.get("pre_decision_activities", [])
    post_decision_0_list = preprocess_config.get("post_decision_0_activities", [])
    post_decision_1_list = preprocess_config.get("post_decision_1_activities", [])

    pre_decision_set = set(pre_decision_list)
    post_decision_0_set = set(post_decision_0_list)
    post_decision_1_set = set(post_decision_1_list)

    observation_rows = []

    for case_id, group in df_sorted.groupby(case_id_col):
        indices = group.index.tolist()
        observation_count = 0
        has_pending_decision = False
        last_pre_decision_index = None

        for i, idx in enumerate(indices):
            activity = df_sorted.loc[idx, activity_col]

            if activity in pre_decision_set:
                has_pending_decision = True
                last_pre_decision_index = i

            elif has_pending_decision and activity in post_decision_0_set:
                observation_count += 1
                obs_rows = df_sorted.iloc[[indices[pi] for pi in range(last_pre_decision_index + 1)]].copy()
                obs_rows['observation'] = observation_count
                obs_rows['target'] = 0
                observation_rows.append(obs_rows)
                has_pending_decision = False
                last_pre_decision_index = None

            elif has_pending_decision and activity in post_decision_1_set:
                observation_count += 1
                obs_rows = df_sorted.iloc[[indices[pi] for pi in range(last_pre_decision_index + 1)]].copy()
                obs_rows['observation'] = observation_count
                obs_rows['target'] = 1
                observation_rows.append(obs_rows)
                has_pending_decision = False
                last_pre_decision_index = None

    if not observation_rows:
        return empty_result(df_sorted)

    return pd.concat(observation_rows, ignore_index=True)
