import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Carregar o dataset
df = pd.read_csv("6.1 - dados_elastic_net_regression.csv")

# 2. Separar X e y
X = df[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']].values
y = df['Target'].values

# 3. Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Criar e treinar o modelo
model = ElasticNet(alpha=1.0, l1_ratio=1.0)
model.fit(X_scaled, y)

# 5. Prever com os dados originais escalados
y_pred = model.predict(X_scaled)

# 6. Visualização do modelo original
plt.scatter(y, y_pred, color='blue', label='Dados de treino')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Linha ideal')
plt.xlabel('Valores reais')
plt.ylabel('Valores previstos')
plt.title('Elastic Net Regression')
plt.legend()
plt.grid(True)
plt.show()

# # 7. Novo dado para prever
# novo_dado = np.array([[10.2, 8.5, 6.7, 12.4, 7.3]])
# novo_dado_scaled = scaler.transform(novo_dado)
# novo_y_pred = model.predict(novo_dado_scaled)

# # 8. Visualização com o novo dado
# plt.figure(figsize=(8,6))
# plt.scatter(y, y_pred, color='blue', label='Treino')
# plt.scatter(novo_y_pred[0], novo_y_pred[0], color='green', s=100, label='Novo dado previsto')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Linha ideal')
# plt.xlabel('Valores reais')
# plt.ylabel('Valores previstos')
# plt.title('Elastic Net Regression - Novo Dado')
# plt.legend()
# plt.grid(True)
# plt.show()

# print(f"Valor previsto para o novo dado: {novo_y_pred[0]:.2f}")
