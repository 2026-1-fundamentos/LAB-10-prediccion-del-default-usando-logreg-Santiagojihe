# flake8: noqa: E501
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""

# pylint: disable=import-outside-toplevel

import gzip
import json
import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def homework():
    """
    En este dataset se desea pronosticar el default (pago) del cliente el próximo
    mes a partir de 23 variables explicativas.
    """
    # -------------------------------------------------------------------------
    # 0. CONFIGURACIÓN DE RUTAS AUTOMÁTICAS
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_path = os.path.join(base_dir, "files", "input", "train_data.csv.zip")
    test_path = os.path.join(base_dir, "files", "input", "test_data.csv.zip")
    models_dir = os.path.join(base_dir, "files", "models")
    output_dir = os.path.join(base_dir, "files", "output")
    
    if not os.path.exists(train_path):
        train_path = "files/input/train_data.csv.zip"
        test_path = "files/input/test_data.csv.zip"
        models_dir = "files/models"
        output_dir = "files/output"

    # -------------------------------------------------------------------------
    # 1. Cargar los datos de entrenamiento y prueba
    # -------------------------------------------------------------------------
    train_df = pd.read_csv(train_path, compression="zip")
    test_df = pd.read_csv(test_path, compression="zip")

    # -------------------------------------------------------------------------
    # 2. Limpiar los datos
    # -------------------------------------------------------------------------
    def clean_data(df):
        df = df.copy()
        df = df.rename(columns={"default payment next month": "default"})
        
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
            
        df = df.dropna()
        df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
        return df

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    # -------------------------------------------------------------------------
    # 3. Pipeline de preprocesamiento y clasificación
    # -------------------------------------------------------------------------
    # Tratamos los históricos de pago como categóricos para maximizar el Balanced Accuracy
    categorical_features = [
        "SEX", "EDUCATION", "MARRIAGE", 
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    ]
    numerical_features = [col for col in X_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", MinMaxScaler(), numerical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            # Modelo estándar sin class_weight para forzar alta Precisión
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # -------------------------------------------------------------------------
    # 4. Optimización usando GridSearchCV (El ajuste clave de precisión)
    # -------------------------------------------------------------------------
    # Al forzar a k a ser muy pequeño (8 a 14), eliminamos el ruido y evitamos falsos positivos.
    param_grid = {
        "feature_selection__k": [8, 10, 12, 14],  
        "classifier__C": [0.1, 1.0, 10.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 5. Guardar el modelo
    # -------------------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    with gzip.open(os.path.join(models_dir, "model.pkl.gz"), "wb") as f:
        pickle.dump(grid_search, f)

    # -------------------------------------------------------------------------
    # 6 y 7. Calcular y guardar métricas y matrices
    # -------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    metrics_train_str = json.dumps({
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
        "f1_score": float(f1_score(y_train, y_train_pred, zero_division=0))
    })

    metrics_test_str = json.dumps({
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0))
    })

    matrix_train_str = json.dumps({
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    })

    matrix_test_str = json.dumps({
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    })

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_train_str + "\n")
        f.write(metrics_test_str + "\n")
        f.write(matrix_train_str + "\n")
        f.write(matrix_test_str + "\n")


if __name__ == "__main__":
    homework()