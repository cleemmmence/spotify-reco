"""
Main script to compare ML models on Spotify dataset
"""

from src.dataloader import load_and_split
from src.models import (
    train_linear_regression,
    train_ridge_regression,
    train_random_forest,
    train_xgboost,
)
from src.evaluation import evaluate_model
from src.plots import plot_actual_vs_pred, plot_residuals


def main():
    print("="*60)
    print("Spotify Popularity Prediction : Model Comparison")
    print("="*60)

    #Load data
    print("\n1. Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_split()

    print("\nData loaded successfully")
    print(f"    Train size: {X_train.shape}")
    print(f"    Test size:  {X_test.shape}")

    #Train models
    print("\n2. Training models...")

    lin_model = train_linear_regression(X_train, y_train)
    ridge_model = train_ridge_regression(X_train, y_train, alpha=1.0)
    rf_model = train_random_forest(X_train, y_train, random_state=42)
    xgb_model = train_xgboost(X_train, y_train, random_state=42)
    print("\nAll models trained successfully.")

    #Evaluate
    print("\n3. Evaluating models...")
    lin_metrics = evaluate_model(lin_model, X_test, y_test, "Linear Regression")
    ridge_metrics = evaluate_model(ridge_model, X_test, y_test, "Ridge Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics =evaluate_model(xgb_model, X_test, y_test, "XGBoost")


    # Conclusion (choose best model based on R2)
    results = {"Linear Regression": lin_metrics["r2"],"Ridge Regression": ridge_metrics["r2"],"Random Forest": rf_metrics["r2"],"XGBoost": xgb_metrics["r2"]}

    winner = max(results, key=results.get)

    print("\n" + "=" * 60)
    print(f"Winner: {winner} (R2 = {results[winner]:.3f})")
    print("=" * 60)

    # Visualizations
    print("\n4. Generating plots...")
    y_pred_lin = lin_model.predict(X_test)
    y_pred_ridge = ridge_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    plot_actual_vs_pred(y_test, y_pred_lin, model_name="Linear Regression")
    plot_actual_vs_pred(y_test, y_pred_ridge, model_name="Ridge Regression")
    plot_actual_vs_pred(y_test, y_pred_rf, model_name="Random Forest")
    plot_actual_vs_pred(y_test, y_pred_xgb, model_name="XGBoost")
    plot_residuals(y_test, y_pred_xgb)

if __name__ == "__main__": 
    main()

