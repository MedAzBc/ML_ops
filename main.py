import argparse
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from model_pipeline import prepare_data, save_model, load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate', 'load'], required=True)
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--test_data', type=str, help='Path to test data')
    parser.add_argument('--save', type=str, default='model.joblib', help='Path to save the model')
    parser.add_argument('--load', type=str, help='Path to load the model')
    parser.add_argument('--pca', type=int, help='Number of PCA components')
    
    args = parser.parse_args()
    print(f'Arguments parsed: {args}')
    
    if args.mode == 'train':
        print("Preparing data...")
        X_train, y_train, X_test, y_test, scaler, pca = prepare_data(args.train_data, args.test_data, args.pca)
        
        print("Training model...")
        model = LinearRegression()
        
        mlflow.set_tracking_uri("http://localhost:5000")  # Set tracking URI
        
        # Create and set the custom experiment
        experiment_name = "AzizBchirExperiment"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id  # Ensure we have the run ID
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Compute regression metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "linear_regression_model")
            
            
            # Register the model
            model_uri = f"runs:/{run_id}/linear_regression_model"
            mlflow.register_model(model_uri, "AzizBchirLinearRegression")  # Register model with a specific name
            
            # Generate and log ROC Curve (for classification tasks)
            if len(np.unique(y_test)) == 2:  # Ensure it's a binary classification task
                fpr, tpr, _ = roc_curve(y_test, predictions)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver Operating Characteristic (ROC) Curve")
                plt.legend(loc="lower right")
                
                roc_path = "roc_curve.png"
                plt.savefig(roc_path)
                plt.close()
                
                mlflow.log_artifact(roc_path)
                mlflow.log_metric("AUC", roc_auc)
            
            print(f"Run ID: {run_id}")  # Print run ID for debugging
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
        
        mlflow.end_run()  # Ensure the run is properly closed
        save_model(model, scaler, pca, args.save)
        print(f"Model saved to {args.save}")
    
    elif args.mode == 'evaluate':
        print("Loading model...")
        model, scaler, pca = load_model(args.load)
        
        print("Preparing test data...")
        _, _, X_test, y_test, _, _ = prepare_data(args.train_data, args.test_data, args.pca)
        
        print("Evaluating model...")
        predictions = model.predict(X_test)
        
        # Compute regression metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
    
    elif args.mode == 'load':
        print(f"Loading model from {args.load}...")
        model, scaler, pca = load_model(args.load)
        print("Model loaded successfully!")

if __name__ == "__main__":
    main()

