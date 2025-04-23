import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

dados = pd.read_csv('8.1 - dados_saude_cardiaca.csv')

# print(dados.head())

x = dados[['Idade', 'Colesterol', 'PressaoSistolica', 'Fumante']]
y = dados['DoencaCardiaca']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelo = LogisticRegression()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Idade'], X_test['Colesterol'], c=y_pred, cmap='coolwarm', marker='o', edgecolor='k', s=100)
plt.title('Resultados da Regressão Logística')
plt.xlabel('Idade')
plt.ylabel('Colesterol')
plt.colorbar(label='Classificação Prevista')
plt.show()