import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

def plot_actual_vs_pred(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Reference line: perfect predictions
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "results",
            f"actual_vs_pred_{model_name.lower().replace(' ', '_')}.png"
        )
    )
    plt.close()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(7,4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors (XGBoost)")
    plt.tight_layout()
    plt.savefig("results/residuals_xgboost.png")
    plt.close()
