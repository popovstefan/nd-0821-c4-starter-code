def encode(df, encoder):
    """Step 2 function, for encoding / preprocessing the data set"""
    df_y = df["exited"]
    df_x = df.drop(["exited"], axis=1)

    categorical_features = ["corporation"]
    df_x_categorical = df_x[categorical_features].values
    df_x_continuous = df_x.drop(*[categorical_features], axis=1)

    if not encoder:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        df_x_categorical = encoder.fit_transform(df_x_categorical)
    else:
        df_x_categorical = encoder.transform(df_x_categorical)
    df_x = np.concatenate([df_x_categorical, df_x_continuous], axis=1)

    return df_x, df_y, encoder
