from flytekit import task, workflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from typing import Tuple

@task
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@task
def train_model(X_train:np.ndarray, y_train:np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

@task
def evaluate_model(model:RandomForestClassifier, X_test:np.ndarray, y_test:np.ndarray) -> float:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

@workflow
def ml_pipeline_wf() -> float:
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train=X_train, y_train=y_train)
    accuracy = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    return accuracy

if __name__ == "__main__":
    print("Pipeline accuracy:", ml_pipeline_wf())
