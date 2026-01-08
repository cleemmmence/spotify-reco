# Spotify Song Popularity Prediction: Model Comparison

## Research Question
To what extent can song popularity be predicted using audio features and basic metadata, and which regression models perform best for this task?

## Setup

## Create environment
conda env create -f environment.yml
conda activate song-popularity

## Usage
python main.py

Expected output : Comparison of regression models using R^2, RMSE, and MAE metrics, as well as saved results in the results/ folder.

## Project Structure
spotify-reco/
├── main.py # Main entry point
├── src/ # Source code
│ ├── dataloader.py # Data loading and preprocessing
│ ├── models.py # Model training
│ ├── evaluation.py # Evaluation metrics
│ └── plots.py # Feature importance plots
├── data/
│ ├── raw/ # Raw datasets
│ └── processed/ # Cleaned dataset used for modeling
├── results/ # Output metrics and figures
│ ├── metrics.csv
│ ├── feature_importance_random_forest.png
│ └── feature_importance_xgboost.png
└── environment.yml # Project dependencies

## Results
Linear Regression Results:
R2 Score: 0.133
RMSE: 21.312
MAE: 17.640

Ridge Regression Results:
R2 Score: 0.133
RMSE: 21.312
MAE: 17.640

Random Forest Results:
R2 Score: 0.175
RMSE: 20.783
MAE: 16.744

XGBoost Results:
R2 Score: 0.181
RMSE: 20.707
MAE: 16.763

Winner : Tree-based models (Random Forest and XGBoost) achieved the highest predictive performance, outperforming linear regression approaches.

## Requirements
- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost