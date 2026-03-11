#evaluate_moddel.py

from .preprocessor import Preprocessor
from .modeling import CreditRiskModel

def evaluate_model(data, target_col, numeric_cols, categorical_cols, yes_no_col):
    # Preprocess the data
    preprocessor = Preprocessor()
    processed_data = preprocessor.preprocess(data, numeric_cols, categorical_cols, yes_no_col)

    # Split features and target
    X, y = preprocessor.split_features_target(processed_data, target_col)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y)

    # Standardize the features
    X_train_scaled, X_test_scaled = preprocessor.standardscaler(X_train, X_test)

    # Train the model
    model = CreditRiskModel()
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

