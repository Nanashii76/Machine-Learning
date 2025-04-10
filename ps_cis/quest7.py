'''
7. Treine um modelo de árvore de decisão para prever a presença de diabetes com
base nas variáveis do dataset. Qual foi a acurácia obtida? Discuta os resultados e
possíveis melhorias para o modelo.
'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier


# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Separando as variáveis independentes (X) e dependentes (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

## Dividir o dataset em conjunto de treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Criar e treinar o modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

## Previsões
y_pred = model.predict(X_test)

## Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))


## Visualizar a árvore de decisão
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Não Diabetes', 'Diabetes'])
plt.title('Árvore de Decisão para Previsão de Diabetes')
plt.show()


