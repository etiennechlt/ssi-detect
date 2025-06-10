import os
import json
import pandas as pd
from models.dl_models import build_dnn_model, cross_validate_dnn, train_and_evaluate_dnn
from models.ml_models import initialize_models, cross_validate_model, train_and_evaluate_model
from preprocessing import split_and_preprocess
import warnings
from utils.metrics import calculate_metrics, cv_metrics_to_df
# from utils.evaluation import save_combined_predictions, save_combined_metrics

warnings.filterwarnings("ignore")

def main(config_path):
    """
    Main training and evaluation pipeline.

    Parameters:
    - config_path: str, path to the configuration file
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_path = os.path.join(os.path.dirname(config_path), config["dataset_path"])
    results_path = os.path.join(os.path.dirname(config_path), config["results_path"])
    artifacts_path = os.path.join(os.path.dirname(config_path), config["artifacts_path"])
    target_column = config.get("target_column", "infection")
    exclude_columns = config.get("exclude_columns", [])
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    normalize = config.get("normalize", True)
    threshold = config.get("threshold", 0.5)
    num_cv_splits = config.get("num_cv_splits", 5)

    # Split and preprocess the dataset
    X_train, X_test, y_train, y_test, scaler = split_and_preprocess(
        dataset_path,
        exclude_columns=exclude_columns,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        normalize=normalize
    )

    # Prepare containers for combined results
    combined_predictions = pd.DataFrame(y_test, columns=["True_Label"])
    combined_metrics = pd.DataFrame()
    cv_results = pd.DataFrame()

    # Cross-validation and evaluation for machine learning models
    ml_models = initialize_models()
    for model_name, model in ml_models.items():
        print(f"Cross-validating {model_name}...")
        cv_metrics = cross_validate_model(model, X_train, y_train, n_splits=num_cv_splits)

        # Store metrics for cross-validation
        cv_results = cv_metrics_to_df(model_name, cv_metrics, cv_results)

        # Train on full training data and evaluate on test set
        print(f"Training {model_name} on full training data...")
        model_file = os.path.join(artifacts_path, model_name + ".pkl")
        ml_metrics, y_test_pred = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_file)

        # Add predictions to combined DataFrame
        combined_predictions[model_name] = y_test_pred

        # Store metrics for combined metrics file
        row_df = pd.DataFrame(ml_metrics, index=[model_name])
        combined_metrics = pd.concat([combined_metrics, row_df], axis=0)

        print(f"{model_name} Metrics: {ml_metrics}")


    # Cross-validation and evaluation for deep learning model
    print("Cross-validating DeepLearningModel...")
    dnn_model = build_dnn_model(input_dim=X_train.shape[1], dropout_rate=0.5, lr=1e-4)
    dnn_cv_metrics = cross_validate_dnn(dnn_model, X_train, y_train, n_splits=num_cv_splits)

    # Store metrics for cross-validation
    cv_results = cv_metrics_to_df("DenseNeuralNet", dnn_cv_metrics, cv_results)

    print("Training DenseNeuralNet on full training data...")
    model_file = os.path.join(artifacts_path, "DenseNeuralNet.h5")
    dnn_metrics, y_test_pred = train_and_evaluate_dnn(dnn_model, X_train, y_train, X_test, y_test, model_file)

    # Add predictions to combined DataFrame
    combined_predictions["DenseNeuralNet"] = y_test_pred

    # Store metrics for combined metrics file
    row_df = pd.DataFrame(dnn_metrics, index=["DenseNeuralNet"])
    combined_metrics = pd.concat([combined_metrics, row_df], axis=0)

    print("\nDenseNeuralNet Results:")
    print(dnn_metrics)

    # Save combined predictions and metrics
    cv_results.to_csv(os.path.join(results_path, "cross_validation_results.csv"), sep=",")
    combined_predictions.to_csv(os.path.join(results_path, "prediction_results.csv"), sep=",")
    combined_metrics.to_csv(os.path.join(results_path, "performance_results.csv"), sep=",")


if __name__ == "__main__":
    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the configuration file
    config_path = os.path.join(root_dir, "../config.json")

    # Call the main function with the configuration file path
    main(config_path)