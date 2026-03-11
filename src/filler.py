import re
import numpy as np
import pandas as pd

class Filler:

    @staticmethod
    def object_turner(data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # Replace "?" with NaN globally
        data.replace("?", np.nan, inplace=True)

        cols = [col for col in data.columns if col != "#"]

        for col in cols:
            if pd.api.types.is_object_dtype(data[col]):

                # If column contains only digits (after removing NaN)
                if data[col].dropna().str.fullmatch(r"\d+(\.\d+)?").all():
                    data[col] = pd.to_numeric(data[col], errors="coerce")

                # If column contains only alphabetic strings
                elif data[col].dropna().str.fullmatch(r"[a-zA-Z]+").all():
                    data[col] = data[col].astype("category")

        return data

    @staticmethod
    def fill_numeric(data: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        data = data.copy()
        for col in numeric_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mean())
        return data

    @staticmethod
    def fill_categorical(data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        data = data.copy()
        for col in categorical_cols:
            if col in data.columns and not data[col].mode().empty:
                data[col] = data[col].fillna(data[col].mode()[0])
        return data

    @staticmethod
    def yes_no_to_binary(data: pd.DataFrame, col: str) -> pd.DataFrame:
        data = data.copy()

        if col in data.columns:
            unique_vals = set(data[col].dropna().str.lower().unique())

            if unique_vals == {"yes", "no"}:
                data[col] = data[col].str.lower().map({"yes": 1, "no": 0})

        return data
    
    @staticmethod
    def ohe(data: pd.DataFrame, col: str) -> pd.DataFrame:
        data = data.copy()

        if col in data.columns and pd.api.types.is_categorical_dtype(data[col]):
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            data.drop(columns=[col], inplace=True)

        return data