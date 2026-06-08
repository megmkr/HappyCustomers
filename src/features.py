from sklearn.preprocessing import StandardScaler

def select_features(df):

    X = df[
        [
            "X1",
            "X3",
            "X5"
        ]
    ]

    y = df["Y"]

    return X, y


"""def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled"""