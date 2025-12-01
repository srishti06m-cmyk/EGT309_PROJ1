from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


TARGET_COL = "Subscription Status"


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the cleaned model_input_table into train/test sets.

    - Maps target 'Subscription Status' from yes/no -> 1/0
    - Drops any weird labels outside yes/no (just in case)
    """
    df = df.copy()

    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
    df = df[df[TARGET_COL].isin(["yes", "no"])]

    y = (df[TARGET_COL] == "yes").astype(int)
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - scales numeric features
    - one-hot encodes categorical features
    """
    numeric_features = ["Age", "Campaign Calls", "Had Previous Contact", "Loan_Count"]
    numeric_features = [c for c in numeric_features if c in X.columns]

    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def _get_class_weights(y_train: pd.Series):
    """Compute imbalance ratio for use in boosting models."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    if pos == 0:
        ratio = 1.0
    else:
        ratio = neg / pos
    return ratio, [1.0, ratio]


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Pipeline]:
    """
    FAST tuning version:
    - Uses RandomizedSearchCV with ~20 trials per model
    - Uses class weights / scale_pos_weight for imbalance
    - Returns best tuned Pipeline per model.
    """
    preprocessor = _build_preprocessor(X_train)

    # imbalance info
    scale_pos_weight, cat_class_weights = _get_class_weights(y_train)

    # Base models with imbalance handling
    base_models: Dict[str, object] = {
        "log_reg": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
            # no class_weight support, it's okay
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        ),
        "lightgbm": LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        ),
        "catboost": CatBoostClassifier(
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
            class_weights=cat_class_weights,
        ),
    }

    param_distributions: Dict[str, Dict[str, list]] = {
        "log_reg": {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l2"],
        },
        "random_forest": {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 4, 5],
        },
        "xgboost": {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample": [0.7, 0.9],
            "clf__colsample_bytree": [0.7, 0.9],
        },
        "lightgbm": {
            "clf__n_estimators": [200, 300, 400],
            "clf__num_leaves": [15, 31, 63],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample": [0.7, 0.9],
            "clf__colsample_bytree": [0.7, 0.9],
        },
        "catboost": {
            "clf__depth": [4, 6, 8],
            "clf__learning_rate": [0.03, 0.1],
            "clf__l2_leaf_reg": [1, 3, 5],
            "clf__iterations": [200, 300, 400],
        },
    }

    trained_models: Dict[str, Pipeline] = {}

    for name, base_clf in base_models.items():
        # Define steps: Preprocess -> SMOTE -> Classifier
        steps = [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=5)), 
            ("clf", base_clf),
        ]
        
        if "class_weight" in base_clf.get_params():
             base_clf.set_params(class_weight=None)
        if "scale_pos_weight" in base_clf.get_params():
             base_clf.set_params(scale_pos_weight=1)

        pipe = Pipeline(steps)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions[name],
            n_iter=20,       
            scoring="average_precision",
            refit="average_precision",
            n_jobs=-1,
            cv=3,
            verbose=1,
            random_state=42,
        )

        search.fit(X_train, y_train)
        best_pipe: Pipeline = search.best_estimator_
        trained_models[name] = best_pipe

    return trained_models


def evaluate_models(
    trained_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Evaluate tuned models on the test set with THRESHOLD OPTIMISATION.

    For each model:
    - Use predict_proba / decision_function to get scores
    - Search thresholds in [0.1, 0.9]
    - Pick threshold that maximises F1
    - Compute accuracy, precision,recall, F1, ROC-AUC with that threshold

    Returns:
        best_model: fitted sklearn Pipeline
        metrics: dict with per-model accuracy, precision, recall, F1, ROC-AUC, best_threshold
    """
    metrics: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_f1_overall = -1.0

    thresholds = np.linspace(0.1, 0.9, 17)

    for name, model in trained_models.items():
        # Get scores
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)

        best_f1 = -1.0
        best_t = 0.5
        best_acc = 0.0
        best_prec=0.0
        best_rec=0.0


        for t in thresholds:
            y_pred_t = (scores >= t).astype(int)
            f1 = f1_score(y_test, y_pred_t)
            acc = accuracy_score(y_test, y_pred_t)
            prec= precision_score(y_test, y_pred_t, )
            rec= recall_score(y_test, y_pred_t)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_acc = acc
                best_prec=prec
                best_rec=rec

        roc = roc_auc_score(y_test, scores)

        metrics[name] = {
            "accuracy": float(best_acc),
            "precision": float(best_prec),
            "recall": float(best_rec),
            "f1": float(best_f1),
            "roc_auc": float(roc),
            "best_threshold": float(best_t),
        }

        if best_f1 > best_f1_overall:
            best_f1_overall = best_f1
            best_name = name

    best_model = trained_models[best_name]

    return best_model, metrics