
'''
1. Existem valores faltantes ou outliers no dataset? Se sim, como você abordaria o
tratamento dessas inconsistências? Explique as técnicas que utilizaria para lidar
com essas questões.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Verificando por valores faltantes ou nulos
colunas_invalidas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

for col in colunas_invalidas:
    zeros = (df[col] == 0).sum()
    print(f"{col}: {zeros} valores iguais a 0")


## Detectando outliers com boxplot
plt.figure(figsize=(15, 8))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f'{col}')
plt.tight_layout()
plt.show()

