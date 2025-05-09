import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from utils.metrics import calculate_metrics


def time_history_callback():
    class TimeHistory(Callback):
        def on_train_begin(self, logs=None):
            self.times = []

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            self.times.append(time.time() - self.epoch_time_start)

    return TimeHistory()

def build_dnn_model(input_dim, lr=1e-4, dropout_rate=0.5):
    model = Sequential(name='DNN_Model')
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )
    return model



def cross_validate_dnn(model, X, y, n_splits=5, random_state=42):
    """
    Perform cross-validation for a deep learning model.

    Parameters:
    - model: Keras model
    - X: array-like, features
    - y: array-like, target labels
    - n_splits: int, number of folds
    - random_state: int, random seed number

    Returns:
    - metrics_list: list of metric dictionaries for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_list = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        y_val_pred = model.predict(X_val).ravel()
        y_val_labels = (y_val_pred >= 0.5).astype(int)

        metrics = calculate_metrics(y_val, y_val_labels, y_val_pred)
        metrics_list.append(metrics)

    return metrics_list


def train_and_evaluate_dnn(model, X_train, y_train, X_test, y_test, model_file):
    """
    Train a model on the full training dataset and evaluate it on the test dataset.

    Parameters:
    - model: Keras model
    - X_train: array-like, training features
    - y_train: array-like, training labels
    - X_test: array-like, test features
    - y_test: array-like, test labels
    - model_name: str, name of the model

    Returns:
    - metrics: dict, evaluation metrics for the test set
    """

    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    y_test_pred = model.predict(X_test).ravel()

    metrics = calculate_metrics(y_test, y_test_pred)

    model.save(model_file, save_format="h5")

    return metrics, y_test_pred