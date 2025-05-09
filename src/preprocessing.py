import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_dataset(file_path, exclude_columns=None, target_column="infection"):
    data = pd.read_csv(file_path, sep=";")

    if exclude_columns:
        data = data.drop(columns=exclude_columns)

    X = data.drop(columns=[target_column])
    y = data[target_column].values

    return X, y

def preprocess_data(X, normalize=True):
    if normalize:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    return X, None

def remove_identifiers(X, identifier_columns):
    return X.drop(columns=identifier_columns, errors="ignore")

def split_and_preprocess(file_path, exclude_columns=None, target_column="infection", test_size=0.2, random_state=42, normalize=True):
    X, y = load_dataset(file_path, exclude_columns, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, scaler = preprocess_data(X_train, normalize)
    X_test = scaler.transform(X_test) if scaler else X_test

    return X_train, X_test, y_train, y_test, scaler
