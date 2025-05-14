import pandas as pd
import matplotlib.pyplot as plt

# Importando dataset de imigrações entre os países dos anos de 1980-2013
df = pd.read_csv("./data/canadian_immegration_data.csv")
# print(df.columns) # Analisando quais colunas estão presentes no dataset

df.set_index('Country', inplace=True) # Criando indexação por países
anos = list(map(str, range(1980, 2014))) # Criando lista de anos de 1980 até 2013

brazil = df.loc["Brazil", anos] # Localiando no dataset todas as ocorrências de Brazil e armazenando o fluxo de imigração de cada ano

# Criando dicionário com resultado da filtragem de Brasil para torná-lo um DataFrame
brazil_dict = {
    'ano': brazil.index.tolist(),
    'imigrantes': brazil.values.tolist(),
}

# Convertendo dicionário para DataFrame
dados_brazil = pd.DataFrame(brazil_dict)
# print(dados_brazil)

# Plotando gráfico anos x imigantes no Brasil
# plt.figure(figsize=(30,8))
plt.plot(dados_brazil['ano'], dados_brazil['imigrantes'])
plt.title("Imigração do Brasil para o Canadá")
plt.xlabel("Ano")
plt.ylabel("Número de Imigrantes")
plt.xticks([x for x in map(str,range(1980,2014,5))])
plt.grid(True)
plt.show()

# Pegano informações de imigração da Argentina
argentina = df.loc["Argentina", anos]

# Criando dicionário com resultado de filtragem de Argentina para torná-lo um Dataframe
argentina_dict = {
    'ano': argentina.index.tolist(),
    'imigrantes': argentina.values.tolist(),
}

# convertendo dicionário para DataFrame
dados_argentina = pd.DataFrame(argentina_dict)

# Plotando gráfico anos x imigrantes (realizando comparativo entre Brasil e Argentina)
plt.plot(dados_brazil['ano'], dados_brazil['imigrantes'], label='Brazil')
plt.plot(dados_argentina['ano'], dados_argentina['imigrantes'], label='Argentina')
plt.title("Comparação de imigração do Brasil e da Argentina para o Canadá")
plt.xlabel("Ano")
plt.ylabel("Número de Imigrantes")
plt.xticks([x for x in map(str, range(1980,2014,5))])
plt.grid(True)
plt.legend()
plt.show()

# Plotando figuras no gráfico
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(dados_brazil['ano'], dados_brazil['imigrantes'])
ax.set_title("Imigração do Brasil para o Canadá\n1980-2013")
ax.set_xlabel("Ano")
ax.set_ylabel("Número de Imigrantes")
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
plt.show()

# Plotando vários gráficos em uma única figura
fig, axs = plt.subplots(1, 2, figsize=(15,5))
axs[0].plot(dados_brazil['ano'], dados_brazil['imigrantes'])
axs[0].set_title("Imigração do Brasil para o Canadá\n1980-2013")
axs[0].set_xlabel("Ano")
axs[0].set_ylabel("Número de Imigrantes")
axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
axs[1].plot(dados_argentina['ano'], dados_argentina['imigrantes'])
axs[1].set_title("Imigração da Argentina para o Canadá\n1980-2013")
axs[1].set_xlabel("Ano")
axs[1].set_ylabel("Número de Imigrantes")
axs[1].xaxis.set_major_locator(plt.MultipleLocator(5))
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15,5))
axs[0].plot(dados_brazil['ano'], dados_brazil['imigrantes'])
axs[0].set_title("Imigração do Brasil para o Canadá\n1980-2013")
axs[0].set_xlabel("Ano")
axs[0].set_ylabel("Número de Imigrantes")
axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
axs[1].boxplot(dados_brazil['imigrantes'])
axs[1].set_title("Boxplot de Imigração do Brasil para o Canadá\n1980-2013")
axs[1].set_ylabel("Número de Imigrantes")
axs[1].set_xlabel("Brasil")
plt.show()