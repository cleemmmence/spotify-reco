from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Train Ridge Regression model.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train Random Forest Regressor.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, random_state=42):
    """
    Train XGBoost Regressor
    """
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

