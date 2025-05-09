import os
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, tree, naive_bayes, discriminant_analysis
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from utils.metrics import calculate_metrics


def initialize_models():
    return {
        "BernoulliNB": naive_bayes.BernoulliNB(alpha=1.0),
        "DecisionTreeClassifier": tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=4),
        "SVC": svm.SVC(probability=True, kernel="rbf", C=1.0),
        "QuadraticDiscriminantAnalysis": discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param=1e-6),
        "XGBClassifier": XGBClassifier(learning_rate=0.1, alpha=1, min_split_loss=0, reg_lambda=3, max_depth=3, n_jobs=1, verbosity=0)
    }



def cross_validate_model(model, X, y, n_splits=5):
    """
    Perform cross-validation for a given model.

    Parameters:
    - model: sklearn-like estimator
    - X: array-like, features
    - y: array-like, target labels
    - n_splits: int, number of folds

    Returns:
    - metrics_list: list of metric dictionaries for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)

        y_val_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)

        metrics = calculate_metrics(y_val, y_val_pred)
        metrics_list.append(metrics)

    return metrics_list


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_file):
    """
    Train a model on the full training dataset and evaluate it on the test dataset.

    Parameters:
    - model: sklearn-like estimator
    - X_train: array-like, training features
    - y_train: array-like, training labels
    - X_test: array-like, test features
    - y_test: array-like, test labels
    - model_name: str, name of the model

    Returns:
    - metrics: dict, evaluation metrics for the test set
    """
    model.fit(X_train, y_train)

    y_test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)

    metrics = calculate_metrics(y_test, y_test_pred)

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    return metrics, y_test_pred
