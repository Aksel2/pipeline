import pandas as pd
import wittgenstein as lw
from imodels import FIGSClassifier, RuleFitClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


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


def train_and_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model


def train_dtc(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    max_depth = model_params.get('max_depth', None)
    criterion = model_params.get('criterion', 'gini')
    class_weight = model_params.get('class_weight', None)

    dtc = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        class_weight=class_weight,
        random_state=42
    )
    return train_and_score(dtc, X_train, X_test, y_train, y_test)


def train_figs(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    max_rules = model_params.get('max_rules', 12)
    figs = FIGSClassifier(max_rules=max_rules, random_state=42)
    return train_and_score(figs, X_train, X_test, y_train, y_test)


def train_ebc(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    ebc = ExplainableBoostingClassifier(random_state=42)
    return train_and_score(ebc, X_train, X_test, y_train, y_test)


def train_rulefit(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    max_rules = model_params.get('max_rules', 30)
    rfc = RuleFitClassifier(max_rules=max_rules, random_state=42)
    return train_and_score(rfc, X_train, X_test, y_train, y_test)


def train_ripper(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    ripper = lw.RIPPER()
    return train_and_score(ripper, X_train, X_test, y_train, y_test)
