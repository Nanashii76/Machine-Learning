import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

'''
    Suponha que temos um conjunto de
    dados de vendas de uma loja,
    incluindo o número de anúncios em
    redes sociais, o número de anúncios
    em jornais e as vendas totais do dia.

    Use regressão linear múltipla para
    prever as vendas do dia com base
    no número de anúncios em redes
    sociais e em jornais.

'''

# Dados de exemplo
X = np.array([[3,5], [2,3], [4,6], [5,2], [3,4], [4,5], [5,3], [6,2], [4,4], [3,6]]).reshape(-1,2)
y = np.array([200,150,250,300,225,275,350,400,275,225])

# Modelo de regressão linear múltipla
model = LinearRegression()

# treinando modelo
model.fit(X, y)

# fazendo previsões com o modelo treinado
y_pred = model.predict(X)

# plotando os dados originais e a linha de regressão
plt.scatter(X[:, 0], y, color='blue', label='Anúncios em redes sociais') # dados originais
plt.scatter(X[:, 1], y, color='green', label='Anúncios em jornais') # dados originais
plt.plot(X[:, 0], y_pred, color='red', label='Linha de regressão') # linha de regressão
plt.title('Relação entre anúncios e vendas')
plt.xlabel('Anúncios')
plt.ylabel('Vendas')
plt.legend()
plt.grid(True)
plt.show()


# Visualização 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Dados Originais')
ax.scatter(X[:, 0], X[:, 1], y_pred, color='red', label='Previsões')

ax.set_xlabel('Anúncios em redes sociais')
ax.set_ylabel('Anúncios em jornais')
ax.set_zlabel('Vendas')
ax.set_title('Regressão Linear Múltipla')
plt.legend()
plt.show()

