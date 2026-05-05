from sklearn.tree import export_text


def extract_ripper_rules(model, feature_names=None):
    rules = ""
    if hasattr(model, 'ruleset_') and model.ruleset_:
        for i, rule in enumerate(model.ruleset_.rules, 1):
            rules += f"Rule {i}: {rule}\n"
    else:
        rules += "No rules learned (all instances classified as default class).\n"

    if hasattr(model, 'classes_'):
        default_class = model.classes_[0] if hasattr(model, 'class_order_') else "unknown"
        rules += f"\nDefault class: {default_class}"

    return rules


def extract_figs_rules(model, feature_names=None):
    try:
        return str(model)
    except Exception:
        return "Unable to extract rules from FIGS model"


def extract_decision_tree_rules(model, feature_names=None):
    if feature_names is not None:
        return export_text(model, feature_names=feature_names)
    return export_text(model)


def extract_ebc_rules(model, feature_names=None):
    try:
        importances = model.term_importances()
        feature_names_model = model.term_names_

        rules = "Feature Importances (Top 10):\n"
        rules += "=" * 50 + "\n"

        importance_pairs = list(zip(feature_names_model, importances))
        importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, importance) in enumerate(importance_pairs[:10], 1):
            rules += f"{i}. {feature}: {importance:.4f}\n"

        return rules
    except Exception as e:
        return f"Unable to extract interpretable components from EBC: {e}"


def extract_rulefit_rules(model, feature_names=None):
    try:
        rules = model._get_rules()
        rules = rules[rules['coef'] != 0].sort_values('importance', ascending=False)

        output = "Weighted Rules:\n"
        output += "=" * 60 + "\n"
        for _, row in rules.iterrows():
            output += f"IF {row['rule']}  weight={row['coef']:.4f}  importance={row['importance']:.4f}\n"

        return output if len(rules) > 0 else "No rules with non-zero coefficients."
    except Exception as e:
        return f"Unable to extract rules from RuleFit model: {e}"


def extract_model_rules(model, model_name, feature_names=None):
    model_name_lower = model_name.lower()

    if 'ripper' in model_name_lower:
        return extract_ripper_rules(model, feature_names)
    elif 'figs' in model_name_lower:
        return extract_figs_rules(model, feature_names)
    elif 'decision_tree' in model_name_lower or 'dtc' in model_name_lower:
        return extract_decision_tree_rules(model, feature_names)
    elif 'ebc' in model_name_lower or 'explainable' in model_name_lower:
        return extract_ebc_rules(model, feature_names)
    elif 'rulefit' in model_name_lower:
        return extract_rulefit_rules(model, feature_names)
    else:
        return f"Rule extraction not implemented for model type: {model_name}"
