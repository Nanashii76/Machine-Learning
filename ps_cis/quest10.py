
'''
10. Utilize regressão logística para estimar a probabilidade de um paciente ser
diagnosticado com diabetes. Quais variáveis são mais influentes no modelo e
como elas impactam a probabilidade de diagnóstico?
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## dividir o dataset em variáveis independentes (X) e dependentes (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

## removendo valores nulos [substituindo pela média]
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Treinando o modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

## Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

## Matriz de Confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

## Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

## Interpretação dos coeficientes
print("\nCoeficientes do modelo:")
print(model.coef_)
print("\nIntercepto do modelo:")
print(model.intercept_)


