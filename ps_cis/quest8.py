
'''
8. A variável DiabetesPedigreeFunction está relacionada à presença de diabetes?
Pacientes com histórico familiar de diabetes apresentam maior risco? Realize
uma análise exploratória e estatística para verificar essa relação.
'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")


# print(df['DiabetesPedigreeFunction'].dtype)
# print(df['DiabetesPedigreeFunction'].head())

## normalizar os valores de DiabetesPedigreeFunction
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'] / 1000
df['DiabetesPedigreeFunction'].describe()

## Boxplot de DiabetesPedigreeFunction por grupo
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=df, palette='Set2')
plt.title('Boxplot de DiabetesPedigreeFunction por Diabetes')
plt.xlabel('Diabetes (0 = Não, 1 = Sim)')
plt.ylabel('DiabetesPedigreeFunction')
plt.grid(True)
plt.show()

## Comparando as médias de DiabetesPedigreeFunction entre os grupos
print("Média de DiabetesPedigreeFunction por Diabetes:")
print(df.groupby('Outcome')['DiabetesPedigreeFunction'].mean())
print("\nDiferença de DiabetesPedigreeFunction entre os grupos:")
print(df.groupby('Outcome')['DiabetesPedigreeFunction'].mean()[1] - df.groupby('Outcome')['DiabetesPedigreeFunction'].mean()[0])
