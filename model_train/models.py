from imodels import FIGSClassifier, C45TreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import wittgenstein as lw


def prepare_features_and_target(df):
    df_clean = df[df['target'].notna()].copy()
    df_clean['target'] = pd.to_numeric(df_clean['target'])

    columns_to_drop = ['target', 'observation']

    for col in ['case_id', 'activity', 'start_time', 'end_time']:
        if col in df_clean.columns:
            columns_to_drop.append(col)

    X = df_clean.drop(columns=columns_to_drop, errors='ignore')
    y = df_clean['target']

    return X, y


def train_dtc(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    print("\n--- Target Distribution ---")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"\nTraining set class distribution:\n{y_train.value_counts()}")
    print(f"Class proportions:\n{y_train.value_counts(normalize=True)}")
    print(f"\nTest set class distribution:\n{y_test.value_counts()}")

    max_depth = model_params.get('max_depth', None)
    criterion = model_params.get('criterion', 'gini')
    class_weight = model_params.get('class_weight', None)

    dtc = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        class_weight=class_weight,
        random_state=42
    )
    dtc.fit(X_train, y_train)

    train_score = dtc.score(X_train, y_train)
    test_score = dtc.score(X_test, y_test)

    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Hyperparameters: max_depth={max_depth}, criterion={criterion}, class_weight={class_weight}")

    return dtc


def train_c45(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    print("\n--- C4.5 Tree Classifier ---")

    c45 = C45TreeClassifier()
    c45.fit(X_train, y_train)

    train_score = c45.score(X_train, y_train)
    test_score = c45.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    return c45


def train_figs(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    print("\n--- FIGS Classifier ---")

    max_rules = model_params.get('max_rules', 12)

    figs = FIGSClassifier(max_rules=max_rules, random_state=42)
    figs.fit(X_train, y_train)

    train_score = figs.score(X_train, y_train)
    test_score = figs.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Hyperparameters: max_rules={max_rules}")

    return figs


def train_ebc(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    print("\n--- Explainable Boosting Classifier ---")

    ebc = ExplainableBoostingClassifier(random_state=42)
    ebc.fit(X_train, y_train)

    train_score = ebc.score(X_train, y_train)
    test_score = ebc.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    return ebc


def train_ripper(X_train, X_test, y_train, y_test, model_params=None):
    if model_params is None:
        model_params = {}

    print("\n--- RIPPER Classifier ---")

    ripper = lw.RIPPER()
    ripper.fit(X_train, y_train)

    train_score = ripper.score(X_train, y_train)
    test_score = ripper.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"\nRules:\n{ripper}")

    return ripper
