
'''
11. Quais técnicas de feature engineering podem ser aplicadas para melhorar a
previsão do diagnóstico de diabetes utilizando modelos de aprendizado de
máquina? Experimente transformar variáveis existentes, criar novas variáveis a
partir de combinações ou interações e utilize técnicas como encoding,
normalização ou transformação de características. Avalie o impacto dessas
mudanças no desempenho de um modelo de aprendizado de máquina (por
exemplo, Random Forest ou XGBoost).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("diabetes.csv")

## Criar uma nova variável de IMC (Índice de Massa Corporal) por idade
df['BMI_Age'] = df['BMI'] * df['Age']

## Separar as variáveis independentes (X) e dependentes (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

## Verificar se há valores nulos e substitui-los pela média
X = X.fillna(X.median())

## Treinar modelo com random forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

## Previsões
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

## Importancia das variáveis em um gráfico
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns[indices]
importances = importances[indices]
plt.figure(figsize=(12, 6))
plt.title("Importância das Variáveis")
plt.bar(range(X.shape[1]), importances, align="center")
plt.xticks(range(X.shape[1]), features, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

