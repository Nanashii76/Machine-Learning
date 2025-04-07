## Codando exemplo 3.0

``` python
import numpy as np
import matpltlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de exemplo
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,3,6,8,10,11,14,15,17.5,20])

# Modelo de regressão linear
modelo = LinearRegression()

# Treinando modelo com os dados (hora, por nota)
modelo.fit(x,y)

# Fazendo previsões com o modelo treinado
y_pred = modelo.predict(x)

# Plotando os dados originais e a linha de regressão
plt.scatter(x, y, color='blue') # Dados originais
plt.plot(x, y_pred, color='red') # Linha de regressão
plt.title('Relação entre horas de estudo e notas')
plt.xlabel('Horas de Estudo')
plt.ylabel('Notas')
plt.show()

```