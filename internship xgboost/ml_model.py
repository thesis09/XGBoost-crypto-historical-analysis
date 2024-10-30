
### File: ml_model.py ###
### File: ml_model.py ###

import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import joblib


# Function to preprocess data
def preprocess_data(data, features, target):
    X = data[features].dropna()
    y = data.loc[X.index, target]

    # Handle any remaining NaN values - replace them with mean (or you can use other methods)
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train XGBoost model with cross-validation and hyperparameter tuning
def train_xgboost(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 10, 15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    xgb = XGBRegressor()
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error',
                               verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    metrics = {
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred)
    }
    return best_model, metrics
