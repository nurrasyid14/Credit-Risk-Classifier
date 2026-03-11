#preprocessor.py

import re
import numpy as np
import pandas as pd
from .filler import Filler

class Preprocessor:

    @staticmethod
    def preprocess(data: pd.DataFrame,
                   numeric_cols: list,
                   categorical_cols: list,
                   yes_no_col: str) -> pd.DataFrame:

        data = Filler.object_turner(data)
        data = Filler.fill_numeric(data, numeric_cols)
        data = Filler.fill_categorical(data, categorical_cols)
        data = Filler.yes_no_to_binary(data, yes_no_col)
        data = pd.get_dummies(data, columns=categorical_cols)
        return data


    @staticmethod
    def split_X_y(data: pd.DataFrame, target_col: str):
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        X = data.drop(columns=[target_col])
        y = data[target_col]
        return X, y


    @staticmethod
    def train_test_split_data(X, y, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


    @staticmethod
    def apply_standard_scaler(X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
