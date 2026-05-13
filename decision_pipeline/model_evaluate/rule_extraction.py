from sklearn.tree import export_text


def extract_ripper_rules(model):
    rules = ""
    if hasattr(model, 'ruleset_') and model.ruleset_:
        for i, rule in enumerate(model.ruleset_.rules, 1):
            rules += f"Rule {i}: {rule}\n"
    else:
        rules += "No rules learned.\n"

    pos_class = getattr(model, 'pos_class', None)
    if pos_class is not None:
        try:
            default_class = 0 if int(pos_class) == 1 else 1
        except (TypeError, ValueError):
            default_class = f"not {pos_class}"
        rules += f"\nDefault class: {default_class}"

    return rules


def extract_figs_rules(model):
    return str(model)


def extract_decision_tree_rules(model):
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is not None:
        return export_text(model, feature_names=list(feature_names))
    return export_text(model)


def extract_ebc_rules(model):
    importances = model.term_importances()
    feature_names_model = model.term_names_

    rules = "Feature Importances:\n"

    importance_pairs = list(zip(feature_names_model, importances))
    importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    for i, (feature, importance) in enumerate(importance_pairs, 1):
        rules += f"{i}. {feature}: {importance:.4f}\n"

    return rules


def extract_rulefit_rules(model):
    rules = model._get_rules()
    rules = rules[rules['coef'] != 0].sort_values('importance', ascending=False)

    if len(rules) == 0:
        return "No rules with non-zero coefficients."

    output = ""
    for _, row in rules.iterrows():
        output += f"IF {row['rule']}  weight={row['coef']:.4f}  importance={row['importance']:.4f}\n"

    return output


def extract_skope_rules(model):
    if not hasattr(model, 'rules_') or not model.rules_:
        return "No rules learned."

    output = ""
    for i, rule in enumerate(model.rules_, 1):
        if isinstance(rule, tuple):
            rule_str = rule[0]
            stats = rule[1] if len(rule) > 1 else None
            if stats and len(stats) >= 2:
                output += f"Rule {i}: {rule_str}  (precision={stats[0]:.3f}, recall={stats[1]:.3f})\n"
            else:
                output += f"Rule {i}: {rule_str}\n"
        elif hasattr(rule, 'rule'):
            output += f"Rule {i}: {rule.rule}\n"
        else:
            output += f"Rule {i}: {rule}\n"
    return output


EXTRACTORS = {
    "decision_tree": extract_decision_tree_rules,
    "ripper": extract_ripper_rules,
    "figs": extract_figs_rules,
    "rulefit": extract_rulefit_rules,
    "skope_rules": extract_skope_rules,
    "ebc": extract_ebc_rules,
}


def extract_model_rules(model, model_name):
    return EXTRACTORS[model_name](model)
