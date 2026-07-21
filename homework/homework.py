
    # flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Salve el modelo como "files/models/model.pkl".
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(*parts):
    """Construye una ruta absoluta a partir de la raíz del proyecto."""
    return os.path.join(ROOT_DIR, *parts)


def clean_dataframe(df):
    """Paso 1: Limpieza del dataset."""
    df = df.copy()

    # Renombrar columna objetivo y remover ID
    if "default payment next month" in df.columns:
        df.rename(columns={"default payment next month": "default"}, inplace=True)
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Eliminar registros con información no disponible (0 en EDUCATION o MARRIAGE)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]

    # Para EDUCATION, valores > 4 pasan a la categoría 4 ('others')
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    return df


def load_data():
    """Paso 1: Carga y limpia los conjuntos de entrenamiento y prueba."""
    train_df = pd.read_csv(
        _path("files", "input", "train_default_of_credit_card_clients.csv")
    )
    test_df = pd.read_csv(
        _path("files", "input", "test_default_of_credit_card_clients.csv")
    )

    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)

    return train_df, test_df


def split_data(train_df, test_df):
    """Paso 2: Divide en x_train, y_train, x_test, y_test."""
    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    return x_train, y_train, x_test, y_test


def make_pipeline(x_train):
    """Paso 3: Crea el pipeline con OneHotEncoder, MinMaxScaler, SelectKBest y LogisticRegression."""
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [
        col for col in x_train.columns if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", MinMaxScaler(), numerical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("select_k_best", SelectKBest(score_func=f_classif)),
            (
                "regressor",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )

    return pipeline


def make_grid_search(pipeline):
    """Paso 4: Búsqueda de hiperparámetros usando validación cruzada con 10 splits."""
    param_grid = {
        "select_k_best__k": range(1, 25),
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return grid_search


def save_model(model):
    """Paso 5: Guarda el modelo como files/models/model.pkl."""
    os.makedirs(_path("files", "models"), exist_ok=True)
    with open(_path("files", "models", "model.pkl"), "wb") as file:
        pickle.dump(model, file)


def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test):
    """Pasos 6 y 7: Calcula métricas y matrices de confusión y las guarda en metrics.json."""
    os.makedirs(_path("files", "output"), exist_ok=True)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Paso 6: Métricas
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    }

    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    }

    # Paso 7: Matrices de Confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    train_cm_matrix = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }

    test_cm_matrix = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }

    with open(_path("files", "output", "metrics.json"), "w", encoding="utf-8") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")
        file.write(json.dumps(train_cm_matrix) + "\n")
        file.write(json.dumps(test_cm_matrix) + "\n")


def main():
    train_df, test_df = load_data()
    x_train, y_train, x_test, y_test = split_data(train_df, test_df)

    pipeline = make_pipeline(x_train)
    model = make_grid_search(pipeline)
    model.fit(x_train, y_train)

    save_model(model)
    calculate_and_save_metrics(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()