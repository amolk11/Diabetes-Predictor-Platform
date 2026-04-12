"""Complete Training Pipeline with MLflow and Best Model Selection (DVC Compatible)"""

import logging
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple
import warnings
import os

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:

    def __init__(self):
        # ✅ Separate folders (IMPORTANT for DVC)
        self.scaler_dir = Path("models/scaler")
        self.trained_dir = Path("models/trained")
        self.metrics_dir = Path("models/metrics")

        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.all_metrics = {}
        self.best_model_name = None
        self.best_model = None

    def load_data(self):
        df = pd.read_csv("data/diabetes.csv")
        logger.info(f"Data loaded: {df.shape}")
        return df

    def prepare_data(self, df):
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # ✅ Save scaler (fixed path)
        scaler_path = self.scaler_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Scaler saved to {scaler_path}")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def get_models(self):
        return {
            "logistic": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(),
            "gb": GradientBoostingClassifier(),
            "xgb": xgb.XGBClassifier(eval_metric="logloss"),
            "lgb": lgb.LGBMClassifier(),
            "svm": SVC(probability=True)
        }

    def calculate_metrics(self, y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred)
        }

    def train_model(self, model):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cross_val_score(model, self.X_train, self.y_train, cv=cv)

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        metrics = self.calculate_metrics(self.y_test, y_pred, y_proba)
        return model, metrics

    def run(self):
        df = self.load_data()
        self.prepare_data(df)

        models = self.get_models()

        mlflow.set_experiment("diabetes-pipeline")

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                trained_model, metrics = self.train_model(model)

                # Save model
                model_path = self.trained_dir / f"{name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(trained_model, f)

                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                mlflow.sklearn.log_model(trained_model, name)

                self.all_metrics[name] = metrics

                logger.info(f"{name}: {metrics}")

        self.select_best_model()
        self.save_metrics()

    def select_best_model(self):
        best = max(self.all_metrics, key=lambda x: self.all_metrics[x]["f1_score"])

        best_model_path = self.trained_dir / f"{best}.pkl"
        with open(best_model_path, "rb") as f:
            model = pickle.load(f)

        final_path = self.trained_dir / "best_model.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Best model: {best}")

    def save_metrics(self):
        path = self.metrics_dir / "metrics_comparison.json"
        with open(path, "w") as f:
            json.dump(self.all_metrics, f, indent=2)

        logger.info(f"Metrics saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    pipeline = TrainingPipeline()

    if args.prepare:
        df = pipeline.load_data()
        pipeline.prepare_data(df)
    elif args.train:
        pipeline.run()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
