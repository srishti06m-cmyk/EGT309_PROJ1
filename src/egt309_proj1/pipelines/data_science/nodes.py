import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["Subscription Status"]
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
        Trained rf_model.
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_GradientBoostingClassifier(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """Trains the Gradient Boosting Classifier model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.

    Returns:
        Trained gb_model.
    """
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    return gb_model

def train_LogisticRegression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains the Logistic Regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.

    Returns:
        Trained lr_model.
    """
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    return lr_model

def compute_pos_class_weight(y: pd.Series) -> float:
    """Computes the positive class weight for handling class imbalance.

    Args:
        y: Series containing the target variable.

    Returns:
        Positive class weight.
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    pos_class_weight = neg_count / pos_count if pos_count != 0 else 1.0
    return pos_class_weight

def train_XGBClassifier(X_train: pd.DataFrame, y_train: pd.Series, pos_class_weight: float) -> XGBClassifier:
    """Trains the XGBoost Classifier model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.
        pos_class_weight: Weight for the positive class to handle class imbalance.

    Returns:
        Trained xgb_model.
    """
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_class_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_LGBMClassifier(X_train: pd.DataFrame, y_train: pd.Series, pos_class_weight: float) -> LGBMClassifier:
    """Trains the LightGBM Classifier model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.
        pos_class_weight: Weight for the positive class to handle class imbalance.

    Returns:
        Trained lgbm_model.
    """
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_class_weight,
        random_state=42
    )
    lgbm_model.fit(X_train, y_train)
    return lgbm_model

def train_CatBoostClassifier(X_train: pd.DataFrame, y_train: pd.Series, pos_class_weight: float) -> CatBoostClassifier:
    """Trains the CatBoost Classifier model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for subscription status.
        pos_class_weight: Weight for the positive class to handle class imbalance.

    Returns:
        Trained catboost_model.
    """
    catboost_model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        scale_pos_weight=pos_class_weight,
        random_seed=42,
        verbose=0
    )
    catboost_model.fit(X_train, y_train)
    return catboost_model

def evaluate_MachineLearningModels(
    rf_model, gb_model, lr_model, xgb_model, lgbm_model, catboost_model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model") :
    """Evaluating all Machine Learning models based on accuracy, precision, recall and F1-score.

    Args:
        classifier: Trained all 6 models.
        X_test: Testing data of independent features.
        y_test: Testing data subscription status.
    """
    models = {
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "Logistic Regression": lr_model,
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
        "CatBoost": catboost_model
    }
    results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        logger = logging.getLogger(__name__)
        logger.info(f"{model_name} Accuracy: {accuracy:.3f}")
        logger.info(f"{model_name} Precision: {precision:.3f}")
        logger.info(f"{model_name} Recall: {recall:.3f}")   
        logger.info(f"{model_name} F1-Score: {f1:.3f}")
        
        results[model_name] = {     
            "model" : model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    return results