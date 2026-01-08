"""
Model evaluation and visualization.
"""

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
import numpy as np


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate regression model and print results.
    """
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"R2 Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")

    return {
        "model": model_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }