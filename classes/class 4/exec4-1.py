import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# lê o arquivo csv
df = pd.read_csv('4.1 - dados_bicicletas.csv')

# print(df[['Temperatura_Media', 'Bicicletas_Alugadas']].head())

# separa os dados em variáveis independentes e dependentes
x = df['Temperatura_Media'].values.reshape(-1, 1)
y = df['Bicicletas_Alugadas'].values

# adiciona a colina de 1s para o intercepto
x_b = np.c_[np.ones((x.shape[0], 1)), x]

# coeficiente beta 
beta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print('Coeficientes: ', beta)

# previsões
y_pred = x_b.dot(beta)

plt.scatter(x, y, color='blue', label='Dados Originais')
plt.plot(x, y_pred, color='red', label='Ajuste Linear')
plt.xlabel('Temperatura Média')
plt.ylabel('Bicicletas Alugadas')
plt.title('Relação entre Temperatura Média e Bicicletas Alugadas')
plt.legend()
plt.grid(True)
plt.show()

