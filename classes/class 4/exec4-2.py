import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Lê o arquivo csv
df = pd.read_csv('4.2 - dados_aluguel_bicicletas.csv')

# Separando os dados em variáveis independentes e dependentes
x = df[['Temperatura', 'VelocidadeVento', 'Precipitacao']].values.reshape(-1, 3)
y = df['BicicletasAlugadas'].values

# adiciona a coluna de 1s para o intercepto
x_b = np.c_[np.ones((x.shape[0], 1)), x]

# Coeficiente beta
beta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

# previsões
y_pred = x_b.dot(beta)

# Coeficientes
print('Coeficientes: ', beta)

# Gráficos de dispersão para cada variável explicativa
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

variaveis = ['Temperatura', 'VelocidadeVento', 'Precipitacao']
for i, var in enumerate(variaveis):
    axs[i].scatter(df[var], y, alpha=0.7)
    axs[i].set_xlabel(var)
    axs[i].set_ylabel('Bicicletas Alugadas')
    axs[i].set_title(f'{var} vs Bicicletas Alugadas')
    axs[i].grid(True)

plt.tight_layout()
plt.show()