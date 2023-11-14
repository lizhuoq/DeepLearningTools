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


def plot_corr(df, columns):
    corr_matrix = df[columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='Blues', fmt='.2f', linewidths=1, square=True, annot_kws={"size": 9})
    plt.title('Correlation Matrix', fontsize=15)
    plt.show()


def plot_train_test_dis(df_train, df_test, numeric_columns):
    fig, axes = plt.subplots(len(numeric_columns), 3 ,figsize = (16, len(numeric_columns) * 4.2),
                         gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]})

    for i, col in tqdm(enumerate(numeric_columns)):
        ax = axes[i, 0]
        sns.kdeplot(data=df_train, x=col, ax=ax, linewidth = 2.1, label='Train')
        sns.kdeplot(data=df_test, x=col, ax=ax, linewidth = 2.1, label='Test')
        ax.set_title(f"\n{col}",fontsize = 9, fontweight= 'bold')
        ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75)
        ax.set(xlabel = '', ylabel = '')
        ax.legend()

        ax = axes[i, 1]
        sns.boxplot(data=df_train, y=col, width = 0.25,saturation = 0.90, linewidth = 0.90, fliersize= 2.25, color = '#037d97',
                ax = ax)
        ax.set(xlabel = '', ylabel = '')
        ax.set_title(f"Train",fontsize = 9, fontweight= 'bold')

        ax = axes[i, 2]
        sns.boxplot(data=df_test, y = col, width = 0.25, fliersize= 2.25,
                saturation = 0.6, linewidth = 0.90, color = '#E4591E',
                ax = ax)
        ax.set(xlabel = '', ylabel = '')
        ax.set_title(f"Test",fontsize = 9, fontweight= 'bold')

    plt.suptitle(f"\nDistribution analysis- continuous columns\n",fontsize = 12, fontweight= 'bold',
             y = 0.89, x = 0.57)
    plt.tight_layout()
    plt.show()
