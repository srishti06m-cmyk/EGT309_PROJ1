import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["Subsciption Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_RandomForestClassifier(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Trains the Random Forest Classifier model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.

    Returns:
        Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

def train_GradientBoostingClassifier(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """Trains the Gradient Boosting Classifier model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.

    Returns:
        Trained model.
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_LogisticRegression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains the Logistic Regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.

    Returns:
        Trained model.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_MachineLearningModels(
    model : ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model") -> dict[str, float]:
    """Evaluating Machine Learning models based on accuracy, precision, recall and F1-score.

    Args:
        classifier: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data subscription status.
    """
    if model_name is None:
        model_name = type(model).__name__

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger = logging.getLogger(__name__)
    logger.info("Accuracy: %.3f",model_name, accuracy)
    logger.info("Precision: %.3f",model_name, precision)
    logger.info("Recall: %.3f",model_name, recall)
    logger.info("F1-score: %.3f",model_name, f1)
    return {    
        "model" : model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }