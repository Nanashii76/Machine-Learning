import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
    Suponha que temos um conjunto de dados
    de alunos com as seguintes horas de estudo
    e suas respectivas notas

    Use regressão linear para prever a nota do
    teste com base nas horas de estudo.
'''

# Dados de exemplo
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([50,55,60,65,70,75,80,85,90,95])

# modelo de regressão linear
model = LinearRegression()

# treinando modelo
model.fit(X, y)

# fazendo previsões com o modelo treinado
y_pred = model.predict(X)

# plotando os dados originais e a linha de regressão
plt.scatter(X, y, color='blue') # dados originais
plt.plot(X, y_pred, color='red') # linha de regressão
plt.title('Relação entre horas de estudo e notas')
plt.xlabel('Horas de Estudo')
plt.ylabel('Notas')
plt.show()
