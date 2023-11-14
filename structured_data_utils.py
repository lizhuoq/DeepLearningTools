import pandas as pd

from package import *


def drop_invalid_columns(dataframes: pd.DataFrame):
    df = dataframes.copy()
    invalid_cols = []
    for col in df.columns.to_list():
        percentage = df[col].value_counts(normalize=True, dropna=False).values[0]
        if percentage > 0.99:
            invalid_cols.append(col)

    df.drop(invalid_cols, axis=1, inplace=True)
    return df


def adversial_validation(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame, time_budget: int):
    df_train_adv = train_dataframe.copy()
    df_test_adv = test_dataframe.copy()
    df_train_adv['Is_test'] = 0
    df_test_adv['Is_test'] = 1
    df_adv = pd.concat([df_train_adv, df_test_adv], ignore_index=True)

    automl = AutoML()
    automl_settings = {
        'time_budget': time_budget,
        'metric': 'accuracy',
        'task': 'classification',
        'eval_method': 'cv',
        'n_jobs': -1
    }

    automl.fit(
        X_train=df_adv.drop('Is_test', axis=1),
        y_train=df_adv['Is_test'],
        **automl_settings
    )

    preds_adv = automl.predict_proba(df_adv.drop('Is_test', axis=1))[:, 1]

    return preds_adv
