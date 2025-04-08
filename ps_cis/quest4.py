'''
4. Quais variáveis apresentam maior correlação com a presença de diabetes? Quais
variáveis parecem ser as mais indicativas da presença de diabetes?
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Verificando correlação entre variáveis
corr = df.corr()

## plotando um heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlação entre variáveis')
plt.show()

## mostra as variáveis mais indicativas da presença de diabetes
print("Variáveis mais indicativas da presença de diabetes:")
correlacao_diabetes = corr['Outcome'].sort_values(ascending=False)
print(correlacao_diabetes)