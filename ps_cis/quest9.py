
'''
9. Pacientes com mais de 50 anos têm taxas de diabetes mais altas do que
pacientes mais jovens? Utilize estatísticas descritivas e gráficos comparativos
para demonstrar as diferenças entre esses dois grupos etários.
'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Criar um grupo de pacientes com mais de 50 anos
df['AgeGroup'] = df['Age'].apply(lambda x: 'Mais de 50' if x > 50 else 'Menos de 50')

## Taxa de diabetes por grupo etário
df['DiabetesRate'] = df.groupby('AgeGroup')['Outcome'].transform(lambda x: x.mean())

print("Taxa de diabetes por grupo etário:")
print(df.groupby('AgeGroup')['DiabetesRate'].mean())

## Gráfico de barras comparando as taxas de diabetes entre os grupos etários
sns.barplot(x='AgeGroup', y='DiabetesRate', data=df, palette='Set2')
plt.title('Taxa de Diabetes por Grupo Etário')
plt.xlabel('Grupo Etário')
plt.ylabel('Taxa de Diabetes')
plt.grid(True)
plt.show()
