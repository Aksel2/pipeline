import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def collapse_observations(df, config):
    encoding_config = config.get("encoding_config", {})
    encoding_strategies = encoding_config.get("encoding_strategies", {})
    aggregation_columns = [col.lower() for col in encoding_strategies.get("aggregation", [])]

    preprocess_config = config.get("preprocess_config", {})
    column_names = preprocess_config.get("column_names", {})
    case_id_col = column_names.get("case_id", "case_id")

    if case_id_col not in df.columns:
        matching = df.columns[df.columns.str.lower() == 'case_id']
        if len(matching) == 0:
            raise ValueError("DataFrame must contain a 'case_id' column")
        case_id_col = matching[0]

    if not aggregation_columns:
        return df.groupby(
            [case_id_col, 'observation'],
            sort=False
        ).last().reset_index()

    case_attributes = [attr.lower() for attr in encoding_config.get("case_attributes", [])]
    event_attributes_discrete = [attr.lower() for attr in encoding_config.get("event_attributes_discrete", [])]
    event_attributes_continuous = [attr.lower() for attr in encoding_config.get("event_attributes_continuous", [])]

    for col in aggregation_columns:
        if col in case_attributes:
            raise ValueError(
                f"Column '{col}' is a case attribute and cannot be used with aggregation. "
                f"Only continuous event attributes are allowed."
            )
        if col in event_attributes_discrete:
            raise ValueError(
                f"Column '{col}' is a discrete event attribute and cannot be used with aggregation. "
                f"Only continuous event attributes are allowed."
            )
        if col not in event_attributes_continuous:
            raise ValueError(
                f"Column '{col}' is not in event_attributes_continuous and cannot be used with aggregation."
            )

    group_cols = [case_id_col, 'observation']

    agg_dict = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if col.lower() in aggregation_columns:
            agg_dict[col] = 'sum'
        else:
            agg_dict[col] = 'last'

    return df.groupby(group_cols, sort=False).agg(agg_dict).reset_index()


class DataFrameEncoder:
    def __init__(self, config):
        self.config = config
        encoding_config = config.get("encoding_config", {})

        encoding_strategies = encoding_config.get("encoding_strategies", {})
        self.onehot_columns = [col.lower() for col in encoding_strategies.get("one-hot", [])]
        self.target_based_columns = [col.lower() for col in encoding_strategies.get("target-based", [])]

        case_attributes = [attr.lower() for attr in encoding_config.get("case_attributes", [])]
        event_attributes_discrete = [attr.lower() for attr in encoding_config.get("event_attributes_discrete", [])]

        all_categorical = case_attributes + event_attributes_discrete

        specified_columns = set(self.onehot_columns + self.target_based_columns)
        self.unspecified_categorical = [col for col in all_categorical if col not in specified_columns]

        self.onehot_encoder = None
        self.target_means = {}
        self.global_target_mean = None
        self.fitted_onehot_columns = []

    def fit(self, df):
        df_work = df.copy()

        columns_to_onehot = self.onehot_columns.copy()

        for col in self.unspecified_categorical:
            if col in df_work.columns:
                if df_work[col].dtype == 'object' or df_work[col].dtype.name == 'category':
                    columns_to_onehot.append(col)

        self.fitted_onehot_columns = [col for col in columns_to_onehot if col in df_work.columns]

        if self.fitted_onehot_columns:
            self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            self.onehot_encoder.fit(df_work[self.fitted_onehot_columns])

        if self.target_based_columns and 'target' in df_work.columns:
            df_work['target'] = pd.to_numeric(df_work['target'])
            self.global_target_mean = df_work['target'].mean()

            for col in self.target_based_columns:
                if col in df_work.columns:
                    self.target_means[col] = df_work.groupby(col)['target'].mean().to_dict()

        return self

    def transform(self, df):
        df_encoded = df.copy()

        if self.target_based_columns and 'target' in df_encoded.columns:
            df_encoded['target'] = pd.to_numeric(df_encoded['target'])

            for col in self.target_based_columns:
                if col in df_encoded.columns and col in self.target_means:
                    df_encoded[f'{col}_encoded'] = df_encoded[col].map(self.target_means[col])
                    df_encoded[f'{col}_encoded'] = df_encoded[f'{col}_encoded'].fillna(self.global_target_mean)
                    df_encoded = df_encoded.drop(columns=[col])

        if self.fitted_onehot_columns and self.onehot_encoder is not None:
            missing_cols = [col for col in self.fitted_onehot_columns if col not in df_encoded.columns]
            if missing_cols:
                raise ValueError(f"Columns missing from dataframe that encoder was fit on: {missing_cols}")

            encoded_data = self.onehot_encoder.transform(df_encoded[self.fitted_onehot_columns])
            feature_names = self.onehot_encoder.get_feature_names_out(self.fitted_onehot_columns)
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)
            df_encoded = df_encoded.drop(columns=self.fitted_onehot_columns)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

        return df_encoded

    def fit_transform(self, df):
        return self.fit(df).transform(df)
