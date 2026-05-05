import math
import re
from collections import defaultdict


def extract_tree_metrics(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature

    internal_count = 0
    leaf_depths = []
    feature_levels = defaultdict(list)

    def walk(node_id, depth):
        nonlocal internal_count
        is_leaf = children_left[node_id] == children_right[node_id]

        if is_leaf:
            leaf_depths.append(depth)
        else:
            internal_count += 1
            feature_levels[features[node_id]].append(depth)
            walk(children_left[node_id], depth + 1)
            walk(children_right[node_id], depth + 1)

    walk(0, 0)

    N = internal_count
    D = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0
    DD = sum(max(levels) - min(levels) for levels in feature_levels.values())
    F = len(feature_levels)

    return {"N": N, "D": D, "DD": DD, "F": F}


def extract_figs_tree_metrics(node):
    internal_count = 0
    leaf_depths = []
    feature_levels = defaultdict(list)

    def walk(n, depth):
        nonlocal internal_count
        if n.left is None and n.right is None:
            leaf_depths.append(depth)
        else:
            internal_count += 1
            feature_levels[n.feature].append(depth)
            if n.left is not None:
                walk(n.left, depth + 1)
            if n.right is not None:
                walk(n.right, depth + 1)

    walk(node, 0)

    N = internal_count
    D = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0
    DD = sum(max(levels) - min(levels) for levels in feature_levels.values())
    F = len(feature_levels)

    return {"N": N, "D": D, "DD": DD, "F": F}


def compute_understandability_score(N, D, DD, F, s=28, w1=1, w2=1, w3=1):
    x = w1 * (N + D) + w2 * DD + w3 * F
    understandability = math.exp(-((x / s) ** 2))
    return {
        "understandability": understandability,
        "N": N,
        "D": D,
        "DD": DD,
        "F": F,
        "x": x
    }


def calculate_tree_understandability(model, model_name, s=28, w1=1, w2=1, w3=1):
    model_name_lower = model_name.lower()

    all_metrics = []

    if 'figs' in model_name_lower:
        if hasattr(model, 'trees_'):
            for root_node in model.trees_:
                if hasattr(root_node, 'left') and hasattr(root_node, 'feature'):
                    all_metrics.append(extract_figs_tree_metrics(root_node))
    elif hasattr(model, 'tree_') and hasattr(model.tree_, 'children_left'):
        all_metrics.append(extract_tree_metrics(model.tree_))

    if not all_metrics:
        return None

    if len(all_metrics) == 1:
        m = all_metrics[0]
        return compute_understandability_score(m["N"], m["D"], m["DD"], m["F"], s, w1, w2, w3)

    N = sum(m["N"] for m in all_metrics)
    D = sum(m["D"] for m in all_metrics) / len(all_metrics)
    DD = sum(m["DD"] for m in all_metrics)

    all_features = set()
    for root_node in model.trees_:
        def collect_features(n):
            if n.left is None and n.right is None:
                return
            all_features.add(n.feature)
            if n.left is not None:
                collect_features(n.left)
            if n.right is not None:
                collect_features(n.right)
        collect_features(root_node)
    F = len(all_features)

    return compute_understandability_score(N, D, DD, F, s, w1, w2, w3)


def extract_rules_as_feature_lists(model, model_name):
    model_name_lower = model_name.lower()

    if 'ripper' in model_name_lower:
        return extract_ripper_feature_lists(model)
    elif 'rulefit' in model_name_lower:
        return extract_rulefit_feature_lists(model)

    return None


def extract_ripper_feature_lists(model):
    if not hasattr(model, 'ruleset_') or not model.ruleset_:
        return None

    rules_features = []
    for rule in model.ruleset_.rules:
        if hasattr(rule, 'conds'):
            features = [cond.feature for cond in rule.conds]
            if features:
                rules_features.append(features)

    return rules_features if rules_features else None


def extract_rulefit_feature_lists(model):
    try:
        rules_df = model._get_rules()
        rules_df = rules_df[rules_df['coef'] != 0]
        rules_df = rules_df[rules_df['type'] == 'rule']
    except Exception:
        return None

    if rules_df.empty:
        return None

    rules_features = []
    for _, row in rules_df.iterrows():
        rule_str = row['rule']
        features = re.findall(r'(\S+)\s*[<>=!]+', rule_str)
        if features:
            rules_features.append(features)

    return rules_features if rules_features else None


def extract_ruleset_metrics(rules_as_feature_lists):
    if not rules_as_feature_lists:
        return None

    L = len(rules_as_feature_lists)
    N = sum(len(rule) for rule in rules_as_feature_lists)
    D = N / L if L > 0 else 0

    flattened = []
    for rule in rules_as_feature_lists:
        flattened.extend(rule)

    feature_positions = defaultdict(list)
    for i, feat in enumerate(flattened):
        feature_positions[feat].append(i)

    DD = 0
    for positions in feature_positions.values():
        for i in range(1, len(positions)):
            DD += positions[i] - positions[i - 1]

    F = len(feature_positions)

    return {"N": N, "D": D, "DD": DD, "F": F}


def calculate_ruleset_understandability(model, model_name, s=21, w1=1, w2=1, w3=1):
    rules_features = extract_rules_as_feature_lists(model, model_name)
    if not rules_features:
        return None

    metrics = extract_ruleset_metrics(rules_features)
    if not metrics:
        return None

    return compute_understandability_score(
        metrics["N"], metrics["D"], metrics["DD"], metrics["F"], s, w1, w2, w3
    )
