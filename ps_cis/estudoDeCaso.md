# Questão 1

## Análise Clínica dos Valores

Antes de iniciar o pré-processamento dos dados, é importante entender **quais valores fazem sentido serem 0 ou nulos** no contexto clínico.

| Variável                    | Pode ser zero/nulo? | Justificativa                                                                 |
|-----------------------------|---------------------|-------------------------------------------------------------------------------|
| **Pregnancies**             | ✅ Sim              | Algumas pacientes podem nunca ter engravidado                                 |
| **Glucose**                 | ❌ Não              | Valor 0 indica dado faltante (ninguém vivo tem glicose 0)                     |
| **BloodPressure**           | ❌ Não              | Pressão 0 não é compatível com vida                                           |
| **SkinThickness**           | ⚠️ Duvidoso         | Pode ter valor 0 real, mas geralmente indica valor ausente                   |
| **Insulin**                 | ⚠️ Duvidoso         | 0 pode ser real, mas geralmente representa ausência de medição               |
| **BMI**                     | ❌ Não              | IMC 0 não é possível                                                         |
| **DiabetesPedigreeFunction**| ✅ Sim              | Valor 0 pode significar ausência de histórico familiar                        |
| **Age**                     | ❌ Não              | O dataset é composto por mulheres com mais de 21 anos                        |
| **Outcome**                 | ❌ Não              | É a variável alvo, precisa estar presente para modelagem                     |

---

Com isso, sabemos quais colunas precisam de **tratamento especial** (substituir os 0s por medianas, por exemplo), pois esses zeros são provavelmente dados faltantes e não valores válidos.

Para identificar se há valores faltantes ou nulos vamos utilizar o seguinte algoritmo

``` python

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

```

com base nisso temos o seguinte output:

![quest1-1](./img/quest1-1.PNG)

Como temos valores faltantes, o que poderiamos fazer é substituir os valores faltantes com a média ou a mediana dos valores não-nulos

Para realizar a identificação de outliers, poderiamos utilizar boxplot

``` python


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

```

Dessa forma obtemos esse resultado

![quest1-2](./img/quest1-2.PNG)

O que eu pude entender desse método é que ele calcula inicialmente a mediana dos valores, depois disso divide em outros dois valores Q1 e Q3, o Q1 é um valor onde 25% da amostra é menor que ele, enquanto o Q3, 75% dos valores da amostra são menores que eles, com base nisso calculamos uma relação entre a mediana, o Q1 e Q3, os valores que se distoam dessa relação são os possíveis *outliners*

# Questão 2

Para respresentar os histogramas e os boxplots, utilizaremos a biblioteca seaborn, para pegarmos os histogramas de cada uma das colunas utilizaremos esse algoritmo

``` python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Excluindo coluna outcome
col = df.columns[:-1]

## Histogramas
plt.figure(figsize=(16, 10))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=30, kde=True, color='lightblue')
    plt.title(f'{col}')
plt.tight_layout()
plt.show()

```

com isso obtemos esse resultado:

![quest2-1](./img/quest2-1.png)

Para pegarmos os boxplots vamos usar o mesmo algoritmo da questão passada

``` python

'''
2. Como as principais variáveis (Glucose, BloodPressure, BMI, etc.) estão
distribuídas? Utilize histogramas e boxplots para representar visualmente essas
distribuições e analise as características de cada uma.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Excluindo coluna outcome
col = df.columns[:-1]

## Histogramas
plt.figure(figsize=(16, 10))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=30, kde=True, color='lightblue')
    plt.title(f'{col}')
plt.tight_layout()
plt.show()

## Boxplots
plt.figure(figsize=(16, 10))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f'{col}')
plt.tight_layout()
plt.show()
```

Como resultado:

![quest1-2](./img/quest1-2.PNG)

- **Pregnancies**: Assimétrica à direita, com valores extremos acima de 10 gestações sendo outliers raros, mas possíveis.
- **Glucose**: Distribuição aproximadamente normal, mas com um pequeno pico entre 100–125. Apresenta valores 0 que foram tratados anteriormente.
- **BloodPressure**: Levemente assimétrica para a esquerda, com vários valores 0 (inválidos).
- **SkinThickness** e **Insulin**: Distribuições altamente assimétricas, com muitos zeros e diversos outliers. Isso indica que essas variáveis possuem muitos valores faltantes e dispersos.
- **BMI**: Levemente assimétrica à direita. Também apresenta outliers leves.
- **DiabetesPedigreeFunction**: Distribuição altamente assimétrica, com muitos valores baixos (0.1 a 0.4) e poucos casos muito altos.
- **Age**: Tendência clara à direita, com maioria entre 20 e 40 anos, mas alguns pacientes com mais de 70.

# Questão 3

O boxplot mostra que pacientes com `Outcome = 1` tendem a ter uma **distribuição de idade mais elevada**.

``` python

'''
3. Existe uma correlação entre a idade dos indivíduos e a presença de diabetes?
Realize uma análise estatística (como teste de correlação) e utilize gráficos (como
scatter plot ou boxplot) para ilustrar essa relação.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

## Verificando semelhança entre Idade e Diabetes
sns.boxplot(x='Outcome', y='Age', data=df, palette='Set2')
plt.title('Boxplot de Idade por Diabetes')
plt.xlabel('Diabetes (0 = Não, 1 = Sim)')
plt.ylabel('Idade')
plt.grid(True)
plt.show()
```

Temos como resultado:

![quest3-1](./img/quest3-1.png)

# Questão 4

Para descobrirmos quais variáveis possuem uma correlação maior a ter diabetes, vamos criar um gráfico de heatmap e ordernar os valores, utilizando esse algoritmo:

``` python
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
```

Dessa forma temos esse resultado:

![quest4-1](./img/quest4-1.png)

![quest4-2](./img/quest4-2.PNG)

as variáveis mais relevantes para modelos preditivos de diabetes são `Glucose`, `BMI`, `Age` e `Pregnancies`.

# Questão 5

