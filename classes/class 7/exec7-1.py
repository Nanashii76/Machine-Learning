import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Importando o dataset
df = pd.read_csv('7.0 - dados_rbf_regression.csv')

# print(df.head())

# Separando X e y
X = df[['X']].values
y = df['y'].values

# Separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo SVR com kernel RBF
model = SVR(kernel='rbf', C=100, gamma='scale')
model.fit(X_train, y_train)

# Prever com os dados de teste
y_pred = model.predict(X_test)

# Visualização do modelo
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Dados de Treino')
plt.scatter(X_test, y_test, color='green', label='Dados de Teste')
plt.plot(X_test, y_pred, color='red', label='Modelo RBF')
plt.title('RBF Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

