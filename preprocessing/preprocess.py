import pandas as pd
from pathlib import Path

from preprocessing.bpm_graph import BPMNNodeType, BPMNGraph


def preprocess_event_log_replay(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()

    preprocess_config = config.get("preprocess_config", {})
    column_mapping = preprocess_config.get("column_names", {})

    case_id_col = column_mapping.get("case_id", "case_id")
    activity_col = column_mapping.get("activity", "activity")
    start_time_col = column_mapping.get("start_time", "start_time")
    end_time_col = column_mapping.get("end_time", "end_time")

    bpmn_model_path = preprocess_config["bpmn_model_path"]
    target_gateway_id = preprocess_config["target_gateway_id"]
    outcome_mapping = preprocess_config["outcome_mapping"]

    outcome_mapping = {k: int(v) for k, v in outcome_mapping.items()}

    encoding_config = config.get("encoding_config", {})
    case_attributes = [attr.lower() for attr in encoding_config.get("case_attributes", [])]
    event_attributes_continuous = [attr.lower() for attr in encoding_config.get("event_attributes_continuous", [])]
    event_attributes_discrete = [attr.lower() for attr in encoding_config.get("event_attributes_discrete", [])]

    columns_to_keep = [case_id_col, activity_col, start_time_col, end_time_col]
    for attr in case_attributes + event_attributes_continuous + event_attributes_discrete:
        if attr in df.columns and attr not in columns_to_keep:
            columns_to_keep.append(attr)

    df_sorted = df.sort_values([case_id_col, start_time_col])
    df_sorted = df_sorted[columns_to_keep]

    bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_model_path))

    df_sorted['observation'] = pd.array([pd.NA] * len(df_sorted), dtype=pd.Int64Dtype())
    df_sorted['target'] = pd.array([pd.NA] * len(df_sorted), dtype=pd.Int64Dtype())

    f_arcs_frequency = {}

    bpmn_task_names = set()
    for e_id, e_info in bpmn_graph.element_info.items():
        if e_info.type == BPMNNodeType.TASK:
            bpmn_task_names.add(e_info.name)

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
            for i in range(original_pos + 1):
                idx = indices[i]
                df_sorted.loc[idx, 'observation'] = observation_count
                df_sorted.loc[idx, 'target'] = outcome

    df_sorted = df_sorted[df_sorted["observation"].notna()]

    return df_sorted


def preprocess_event_log(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    df.columns = df.columns.str.lower()

    preprocess_config = config.get("preprocess_config", {})
    column_mapping = preprocess_config.get("column_names", {})

    case_id_col = column_mapping.get("case_id", "case_id")
    activity_col = column_mapping.get("activity", "activity")
    start_time_col = column_mapping.get("start_time", "start_time")
    end_time_col = column_mapping.get("end_time", "end_time")

    pre_decision_activity = preprocess_config.get("pre_decision_activities", [])[0]
    post_decision_0_activity = preprocess_config.get("post_decision_0_activities", [])[0]
    post_decision_1_activity = preprocess_config.get("post_decision_1_activities", [])[0]

    encoding_config = config.get("encoding_config", {})
    case_attributes = encoding_config.get("case_attributes", [])
    event_attributes_continuous = encoding_config.get("event_attributes_continuous", [])
    event_attributes_discrete = encoding_config.get("event_attributes_discrete", [])

    case_attributes = [attr.lower() for attr in case_attributes]
    event_attributes_continuous = [attr.lower() for attr in event_attributes_continuous]
    event_attributes_discrete = [attr.lower() for attr in event_attributes_discrete]

    columns_to_keep = [
        case_id_col,
        activity_col,
        start_time_col,
        end_time_col
    ]

    all_attributes = case_attributes + event_attributes_continuous + event_attributes_discrete
    for attr in all_attributes:
        if attr in df.columns and attr not in columns_to_keep:
            columns_to_keep.append(attr)

    df_sorted = df.sort_values([case_id_col, start_time_col])
    df_sorted = df_sorted[columns_to_keep]

    df_sorted['observation'] = pd.array([pd.NA] * len(df_sorted), dtype=pd.Int64Dtype())
    df_sorted['target'] = pd.array([pd.NA] * len(df_sorted), dtype=pd.Int64Dtype())

    for case_id, group in df_sorted.groupby(case_id_col):
        indices = group.index.tolist()
        observation_count = 0
        last_decision_pos = -1

        for i, idx in enumerate(indices):
            activity = df_sorted.loc[idx, activity_col]

            if activity == pre_decision_activity:
                target_value = None
                for j in range(i + 1, len(indices)):
                    future_idx = indices[j]
                    future_activity = df_sorted.loc[future_idx, activity_col]

                    if future_activity == post_decision_0_activity:
                        target_value = 0
                        break
                    elif future_activity == post_decision_1_activity:
                        target_value = 1
                        break

                if target_value is None:
                    continue

                if last_decision_pos >= 0:
                    for prev_i in range(last_decision_pos + 1):
                        prev_idx = indices[prev_i]
                        df_sorted.loc[prev_idx, 'observation'] = pd.NA
                        df_sorted.loc[prev_idx, 'target'] = pd.NA

                observation_count += 1

                obs_start = last_decision_pos + 1 if last_decision_pos >= 0 else 0
                for prev_i in range(obs_start, i + 1):
                    prev_idx = indices[prev_i]
                    df_sorted.loc[prev_idx, 'observation'] = observation_count
                    df_sorted.loc[prev_idx, 'target'] = target_value

                last_decision_pos = i

    df_sorted = df_sorted[df_sorted["observation"].notna()]

    return df_sorted
