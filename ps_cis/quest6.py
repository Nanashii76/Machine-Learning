
'''
6. Existe um valor específico de glicose que pode ser considerado crítico para o
diagnóstico de diabetes? Utilize gráficos de dispersão e cálculos estatísticos para
investigar esse ponto e definir um limite crítico, se possível.

'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Boxplot de glicose por grupo
sns.boxplot(x='Outcome', y='Glucose', data=df, palette='Set2')
plt.title('Boxplot de Glicose por Diabetes')
plt.xlabel('Diabetes (0 = Não, 1 = Sim)')
plt.ylabel('Glucose')
plt.grid(True)
plt.show()

## Criando gráfico de densidade
sns.kdeplot(df[df['Outcome'] == 0]['Glucose'], label='Sem Diabetes', color='blue')
sns.kdeplot(df[df['Outcome'] == 1]['Glucose'], label='Com Diabetes', color='red')
plt.title('Distribuição de Glicose por Diabetes')
plt.xlabel('Glicose')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.show()

## Médias de glicose por grupo
print("Média de Glicose por Diabetes:")
print(df.groupby('Outcome')['Glucose'].mean())
print("\nDiferença de Glicose entre os grupos:")
print(df.groupby('Outcome')['Glucose'].mean()[1] - df.groupby('Outcome')['Glucose'].mean()[0])

## Criar um limiar crítico
for limite in range(100 ,170,5):
    y_pred_limite = df['Glucose'].apply(lambda x: 1 if x >= limite else 0)
    acc = accuracy_score(df['Outcome'], y_pred_limite)
    print(f"Limiar: {limite}, Acurácia: {acc:.5f}")