import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Importando o dataset
df = pd.read_csv('7.2 - dados_previsao_bicicletas.csv')

# Separando as variáveis independentes e dependentes
X = df[['Temperatura', 'VelocidadeVento', 'Precipitacao']].values
y = df['BicicletasAlugadas'].values

# Separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo SVR com kernel RBF
model = SVR(kernel='rbf', C=100, gamma='scale')
model.fit(X_train, y_train)

# Prever com os dados de teste
y_pred = model.predict(X_test)

# Organiza X_test e y_pred para visualizar melhor a curva
sorted_indices = np.argsort(X_test[:, 0])  # Ordena com base na primeira variável (Temperatura)
X_sorted = X_test[sorted_indices]
y_sorted_pred = y_pred[sorted_indices]

# Novo gráfico mais suave
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, color='blue', label='Treino', alpha=0.6)
plt.scatter(X_test[:, 0], y_test, color='green', label='Teste', alpha=0.6)
plt.plot(X_sorted[:, 0], y_sorted_pred, color='red', label='SVR (RBF)', linewidth=2)
plt.title('Regressão com SVR (Kernel RBF)')
plt.xlabel('Temperatura Média (°C)')
plt.ylabel('Número de Bicicletas Alugadas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



