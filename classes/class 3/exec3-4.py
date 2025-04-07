import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

'''
    Imagine que você é um analista de
    dados em uma empresa imobiliária. A
    empresa deseja prever o preço de
    venda de casas com base em várias
    características da casa.

    Você tem um conjunto de dados que
    inclui o tamanho da casa (em metros
    quadrados), o número de quartos, a
    idade da casa (em anos) e o preço de
    venda (em milhares de reais).

    Use regressão linear múltipla para
    prever o preço de venda das casas
    com base no tamanho, número de
    quartos e idade da casa.
'''

# Dados de exemplo
X = np.array([[120,3,5], [150,4,7], [80, 2, 10], [200,5,3], [100,3,15], [130,3,4], [160, 4, 6], [110, 3, 8], [180, 4, 2], [140,3,12]]).reshape(-1,3)
y = np.array([300, 350, 200, 500, 250, 320, 360, 280, 480, 310])

# Modelo de regressão linear múltipla
model = LinearRegression()

# treinando modelo
model.fit(X, y)

# fazendo previsões com o modelo treinado
y_pred = model.predict(X)

# plotando os dados originais e a linha de regressão
plt.scatter(X[:, 0], y, color='blue', label='Tamanho da casa') # dados originais
plt.scatter(X[:, 1], y, color='green', label='Número de quartos') # dados originais
plt.scatter(X[:, 2], y, color='red', label='Idade da casa') # dados originais
plt.plot(X[:, 0], y_pred, color='black', label='Linha de regressão') # linha de regressão
plt.title('Relação entre características da casa e preço de venda')
plt.xlabel('Características da casa')
plt.ylabel('Preço de venda (milhares de reais)')
plt.legend()
plt.grid(True)
plt.show()

# Visualização 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Dados Originais')
ax.scatter(X[:, 0], X[:, 1], y_pred, color='red', label='Previsões')
ax.set_xlabel('Tamanho da casa (m²)')
ax.set_ylabel('Número de quartos')
ax.set_zlabel('Preço de venda (milhares de reais)')
ax.set_title('Regressão Linear Múltipla')
plt.legend()
plt.show()

