import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate evaluation metrics for a classification model.

    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted probabilities

    Returns:
    - metrics: dict, containing evaluation metric scores
    """

    y_pred_labels = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels, zero_division=0),
        'recall': recall_score(y_true, y_pred_labels, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_labels) if y_pred_labels is not None else None,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0
    }
    return metrics

def cv_metrics_to_df(model_name, metrics_list, results_df):
    df = pd.DataFrame(metrics_list)
    means = df.mean()
    stds = df.std(ddof=1)
    data = {}
    for metric in df.columns:
        data[f"{metric}_mean"] = means[metric]
        data[f"{metric}_std"] = stds[metric]
    return pd.concat([results_df, pd.DataFrame(data, index=[model_name])], axis=0)



def save_combined_predictions(predictions, output_path):
    """
    Save combined predictions for all models to a single CSV file.

    Parameters:
    - predictions: dict, where keys are model names and values are DataFrames with predictions
    - output_path: str, file path to save the combined predictions
    """
    os.makedirs(output_path, exist_ok=True)
    combined_df = pd.concat(predictions.values(), axis=1)
    combined_df.to_csv(os.path.join(output_path, "combined_predictions.csv"), index=False, sep=",")


def save_combined_metrics(metrics, output_path):
    """
    Save combined metrics for all models to a single CSV file.

    Parameters:
    - metrics: dict, where keys are model names and values are metric dictionaries
    - output_path: str, file path to save the combined metrics
    """
    os.makedirs(output_path, exist_ok=True)
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(os.path.join(output_path, "combined_metrics.csv"), sep=",")