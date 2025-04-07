import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
    Imagine que você é um cientista de dados
    trabalhando para uma empresa de aluguel de
    bicicletas. A empresa quer entender como a
    temperatura afeta o número de bicicletas
    alugadas diariamente.

    Você coletou dados sobre a temperatura
    média diária e o número total de bicicletas
    alugadas em vários dias.

    Use regressão linear para prever o número de
    bicicletas alugadas com base na temperatura
    média.
'''

# Dados do exemplo
X = np.array([10,12,14,16,18,20,22,24,26,28]).reshape(-1,1)
y = np.array([100,120,140,160,180,200,220,240,260,280])

# Modelo de regressão linear
model = LinearRegression()

# treinando modelo
model.fit(X, y)

# fazendo previsões com o modelo treinado
y_pred = model.predict(X)

# plotando os dados originais e a linha de regressão
plt.scatter(X, y, color='blue') # dados originais   
plt.plot(X, y_pred, color='red') # linha de regressão
plt.title('Relação entre temperatura e número de bicicletas alugadas')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Número de bicicletas alugadas')
plt.show()