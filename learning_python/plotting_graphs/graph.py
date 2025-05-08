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