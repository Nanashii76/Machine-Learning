
'''
Você é um cientista de dados
encarregado de criar um modelo de
regressão para um conjunto de dados
com múltiplas variáveis independentes.
Seu objetivo é usar
a Elastic Net
Regression, uma técnica que combina
as regularizações L1 e L2, para prever
uma variável dependente a partir de
várias variáveis independentes. Este
método é útil para controlar ooverfitting e lidar com a
multicolinearidade entre as variáveis.
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv("6.1 - dados_elastic_net_regression.csv")
# print(df.head())

## Separar as variáveis independentes (X) e a variável dependente (y)
X = df[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']].values
y = df['Target'].values

## Criar o modelo Elastic Net
model = ElasticNet(alpha=1.0, l1_ratio=1.0)
model.fit(X, y)

## coeficientes do modelo
print("Coeficientes do modelo:")
print(model.coef_)
print("Intercepto do modelo:")
print(model.intercept_)

## Prever os valores de y usando o modelo
y_pred = model.predict(X)

## Plotar os dados e a linha de ajuste
plt.scatter(y, y_pred, color='blue', label='Dados')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Linha de ajuste')
plt.xlabel('Valores reais')
plt.ylabel('Valores previstos')
plt.title('Elastic Net Regression')
plt.legend()
plt.grid(True)
plt.show()

