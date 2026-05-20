# src/features/train_model.py
import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import catboost as catboost
import yaml
import logging
from mlflow.tracking import MlflowClient
import platform
import sklearn

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()

# -----------------------------
# Load model from config
# -----------------------------
def get_model_instance(name, params):
    model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'XGBoost': xgb.XGBRegressor,
        'Catboost': catboost.CatBoostRegressor
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)

# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg['name'])

    # Load data
    data = pd.read_csv(args.data)
    target = model_cfg['target_variable']

    # Use all features except the target variable
    X = data.drop(columns=[target])
    y = data[target]
    # First split: keep test set sealed
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Second split: create validation set from train_val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    logger.info(f"Train Split: {X_train.shape}, Validation Split: {X_val.shape}, Test Split: {X_test.shape}")


    # Get model
    model = get_model_instance(model_cfg['best_model'], model_cfg['parameters'])

    # Start MLflow run
    with mlflow.start_run(run_name="final_training"):
        logger.info(f"Training model: {model_cfg['best_model']}")
        model.fit(X_train, y_train)
        
        # -----------------------------
        # Validation evaluation - log scale
        # -----------------------------
        y_val_pred = model.predict(X_val)

        val_mae = float(mean_absolute_error(y_val, y_val_pred))
        val_mse = float(mean_squared_error(y_val, y_val_pred))
        val_rmse = float(np.sqrt(val_mse))
        val_r2 = float(r2_score(y_val, y_val_pred))
        logger.info("Validation Results - Log Price Scale")
        logger.info(f"Validation R²: {val_r2:.4f}")
        logger.info(f"Validation MAE: {val_mae:.4f}")
        logger.info(f"Validation RMSE: {val_rmse:.4f}")
        logger.info(f"Validation MSE: {val_mse:.4f}")
        # -----------------------------
        # # Test evaluation - log scale
        # # -----------------------------
        y_test_pred = model.predict(X_test)
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        test_mse = float(mean_squared_error(y_test, y_test_pred))
        test_rmse = float(np.sqrt(test_mse))
        test_r2 = float(r2_score(y_test, y_test_pred))
        
        logger.info("Final Test Results - Log Price Scale")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        logger.info(f"Test MSE: {test_mse:.4f}")

        # -----------------------------
        # Validation evaluation - original price scale
        # Dataset only has log_price, so recover price using expm1
        # -----------------------------
        y_val_price = np.expm1(y_val)
        y_val_pred_price = np.expm1(y_val_pred)

        val_mae_price = float(mean_absolute_error(y_val_price, y_val_pred_price))
        val_mse_price = float(mean_squared_error(y_val_price, y_val_pred_price))
        val_rmse_price = float(np.sqrt(val_mse_price))
        val_r2_price = float(r2_score(y_val_price, y_val_pred_price))

        logger.info("Validation Results - Original Price Scale")
        logger.info(f"Validation R² Price Scale: {val_r2_price:.4f}")
        logger.info(f"Validation MAE Price Scale: ${val_mae_price:.2f}")
        logger.info(f"Validation RMSE Price Scale: ${val_rmse_price:.2f}")
        logger.info(f"Validation MSE Price Scale: ${val_mse_price:.2f}")

        # -----------------------------
        # Test evaluation - original price scale
        # Dataset only has log_price, so recover price using expm1
        # -----------------------------
        y_test_price = np.expm1(y_test)
        y_test_pred_price = np.expm1(y_test_pred)

        test_mae_price = float(mean_absolute_error(y_test_price, y_test_pred_price))
        test_mse_price = float(mean_squared_error(y_test_price, y_test_pred_price))
        test_rmse_price = float(np.sqrt(test_mse_price))
        test_r2_price = float(r2_score(y_test_price, y_test_pred_price))

        logger.info("Final Test Results - Original Price Scale")
        logger.info(f"Test R² Price Scale: {test_r2_price:.4f}")
        logger.info(f"Test MAE Price Scale: ${test_mae_price:.2f}")
        logger.info(f"Test RMSE Price Scale: ${test_rmse_price:.2f}")
        logger.info(f"Test MSE Price Scale: ${test_mse_price:.2f}")

        # Log params and metrics
        mlflow.log_params(model_cfg['parameters'])
        mlflow.log_metrics({
            "val_mae_log": val_mae,
            "val_mse_log": val_mse,
            "val_rmse_log": val_rmse,
            "val_r2_log": val_r2,
            
            "test_mae_log": test_mae,
            "test_mse_log": test_mse,
            "test_rmse_log": test_rmse,
            "test_r2_log": test_r2,
            
            "val_mae_price": val_mae_price,
            "val_mse_price": val_mse_price,
            "val_rmse_price": val_rmse_price,
            "val_r2_price": val_r2_price,
            
            "test_mae_price": test_mae_price,
            "test_mse_price": test_mse_price,
            "test_rmse_price": test_rmse_price,
            "test_r2_price": test_r2_price
            })

        # Log and register model
        mlflow.sklearn.log_model(model, "tuned_model")
        model_name = model_cfg['name']
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Transition model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        # Add a human-readable description
        description = (
            f"Model for predicting New York City Airbnb prices.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All features in the dataset except the target variable\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Model saved at: {args.models_dir}/trained/{model_name}.pkl\n"
            f"feature_importance: {dict(zip(X.columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else 'N/A'}\n"
            f"Performance metrics:\n"
            f"  Validation - log scale:\n"
            f"    - MAE: {val_mae:.4f}\n"
            f"    - RMSE: {val_rmse:.4f}\n"
            f"    - MSE: {val_mse:.4f}\n"
            f"    - R²: {val_r2:.4f}\n"
            f"  Test - log scale:\n"
            f"    - MAE: {test_mae:.4f}\n"
            f"    - RMSE: {test_rmse:.4f}\n"
            f"    - MSE: {test_mse:.4f}\n"
            f"    - R²: {test_r2:.4f}\n"
            f"  Validation - price scale:\n"
            f"    - MAE: ${val_mae_price:.2f}\n"
            f"    - RMSE: ${val_rmse_price:.2f}\n"
            f"    - MSE: ${val_mse_price:.2f}\n"
            f"    - R²: {val_r2_price:.4f}\n"
            f"  Test - price scale:\n"
            f"    - MAE: ${test_mae_price:.2f}\n"
            f"    - RMSE: ${test_rmse_price:.2f}\n"
            f"    - MSE: ${test_mse_price:.2f}\n"
            f"    - R²: {test_r2_price:.4f}"
            )
        client.update_registered_model(name=model_name, description=description)

        # Add tags for better organization
        client.set_registered_model_tag(model_name, "algorithm", model_cfg['best_model'])
        client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg['parameters']))
        client.set_registered_model_tag(model_name, "features", "All features except target variable")
        client.set_registered_model_tag(model_name, "target_variable", target)
        client.set_registered_model_tag(model_name, "training_dataset", args.data)
        #client.set_registered_model_tag(model_name, "feature_importance", str(dict(zip(X.columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else 'N/A'))
        client.set_registered_model_tag(model_name, "model_path", f"{args.models_dir}/trained/{model_name}.pkl")

        # Add dependency tags
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "catboost_version": catboost.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in deps.items():
            client.set_registered_model_tag(model_name, k, v)

        # Save model locally
        save_path = f"{args.models_dir}/trained/{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Saved trained model to: {save_path}")
        logger.info(f"Final Test Log-Scale Results | "f"MAE: {test_mae:.4f}, "f"RMSE: {test_rmse:.4f}, "f"R²: {test_r2:.4f}")
        logger.info(f"Final Test Price-Scale Results | "f"MAE: ${test_mae_price:.2f}, "f"RMSE: ${test_rmse_price:.2f}, "f"R²: {test_r2_price:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001
# python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/AB_NYC_2019_feature_engineered.csv   --models-dir models   --mlflow-tracking-uri http://localhost:5001