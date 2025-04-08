
'''
5. Existe uma relação entre o IMC dos pacientes e o diagnóstico de diabetes?
Compare os valores médios de IMC entre os grupos com e sem diabetes, e
analise a diferença estatisticamente.

'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Ver médias de IMC entre os grupos com e sem diabetes
imc_diabetes = df.groupby('Outcome')['BMI'].mean()
print("Média de IMC por Diabetes:")
print(imc_diabetes)
print("\nDiferença de IMC entre os grupos:")
print(imc_diabetes[1] - imc_diabetes[0])

## Boxplot
sns.boxplot(x='Outcome', y='BMI', data=df, palette='Set2')
plt.title('Boxplot de IMC por Diabetes')
plt.xlabel('Diabetes (0 = Não, 1 = Sim)')
plt.ylabel('IMC')
plt.grid(True)
plt.show()
