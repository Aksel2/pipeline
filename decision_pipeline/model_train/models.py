import pandas as pd
import wittgenstein as lw
from imodels import (
    FIGSClassifier,
    RuleFitClassifier,
    SkopeRulesClassifier,
)
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def _build_kwargs(model_params, **defaults):
    kwargs = dict(defaults)
    if model_params:
        for k, v in model_params.items():
            if k == 'enabled':
                continue
            kwargs[k] = v
    return kwargs


def prepare_features_and_target(df, config=None):
    df_clean = df[df['target'].notna()].copy()
    df_clean['target'] = pd.to_numeric(df_clean['target'])

    columns_to_drop = ['target', 'observation']

    if config is not None:
        column_names = config.get("preprocess_config", {}).get("column_names", {})
        for col in column_names.values():
            col_lower = col.lower()
            if col_lower in df_clean.columns and col_lower not in columns_to_drop:
                columns_to_drop.append(col_lower)
    else:
        for col in ['case_id', 'activity', 'start_time', 'end_time']:
            if col in df_clean.columns:
                columns_to_drop.append(col)

    X = df_clean.drop(columns=columns_to_drop, errors='ignore')
    y = df_clean['target']

    for col in X.columns:
        if X[col].isna().any():
            if X[col].dropna().isin([0, 1, True, False, 'True', 'False']).all():
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(X[col].median())

    return X, y


def train_and_score(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def train_dtc(X_train, y_train, model_params=None):
    kwargs = _build_kwargs(
        model_params,
        max_depth=10,
        criterion='gini',
        class_weight='balanced',
        random_state=42,
    )
    return train_and_score(DecisionTreeClassifier(**kwargs), X_train, y_train)


def train_figs(X_train, y_train, model_params=None):
    kwargs = _build_kwargs(
        model_params,
        max_rules=12,
        random_state=42,
    )
    return train_and_score(FIGSClassifier(**kwargs), X_train, y_train)


def train_ebc(X_train, y_train, model_params=None):
    kwargs = _build_kwargs(
        model_params,
        random_state=42,
    )
    return train_and_score(ExplainableBoostingClassifier(**kwargs), X_train, y_train)


def train_rulefit(X_train, y_train, model_params=None):
    kwargs = _build_kwargs(
        model_params,
        max_rules=12,
        n_estimators=20,
        alpha=0.1,
        exp_rand_tree_size=False,
        include_linear=False,
        random_state=42,
    )
    return train_and_score(RuleFitClassifier(**kwargs), X_train, y_train)


def train_ripper(X_train, y_train, model_params=None):
    kwargs = _build_kwargs(
        model_params,
        n_discretize_bins=4,
        max_rule_conds=3,
        dl_allowance=8,
        k=2,
        prune_size=0.33,
        random_state=42,
    )
    return train_and_score(lw.RIPPER(**kwargs), X_train, y_train)


def train_skope(X_train, y_train, model_params=None):
    kwargs = _build_kwargs(
        model_params,
        n_estimators=5,
        max_depth=2,
        precision_min=0.8,
        recall_min=0.05,
        random_state=42,
    )
    return train_and_score(SkopeRulesClassifier(**kwargs), X_train, y_train)
