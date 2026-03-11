#XGBoost.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


class XGBoostModel:
    def __init__(self, params=None):
        self.params = params if params is not None else {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = XGBRegressor(**self.params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

# training & evaluation function
def train_xgboost(X_train, y_train, params=None):
    xgb_model = XGBoostModel(params)
    xgb_model.fit(X_train, y_train)
    return xgb_model


def evaluate_xgboost(model, X_test, y_test):
    mse = model.evaluate(X_test, y_test)
    print(f'XGBoost Mean Squared Error: {mse}')
    return mse

#pickling function
import pickle
def save_xgboost_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        