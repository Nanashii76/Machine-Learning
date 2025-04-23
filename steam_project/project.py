import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Carregar as bases de dados Steam
steam = pd.read_csv('./data/steam.csv')
users = pd.read_csv('./data/steam-200k.csv')
users.columns = ['user_id', 'game', 'action', 'value', 'unused']

'''
    Análise steam.csv (jogos)
'''

print("Ánalise de atributos do dataset steam.csv\n")
jogos_info = []
for col in steam.columns:
    jogos_info.append([col, steam[col].dtype, steam[col].isnull().sum(), steam[col].nunique()])
jogos_info = pd.DataFrame(jogos_info, columns=['coluna', 'tipo', 'nulos', 'unicos'])
print(jogos_info)

# Correlação entre as variáveis numéricas
plt.figure(figsize=(10, 8))
sns.heatmap(steam[['positive_ratings', 'negative_ratings', 'average_playtime', 'price']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlação entre variáveis numéricas')
plt.show()

'''
    Análise steam-200k.csv (usuários)
'''

print("Ánalise de atributos do dataset steam-200k.csv\n")
users_info = []
for col in users.columns:
    users_info.append([col, users[col].dtype, users[col].isnull().sum(), users[col].nunique()])
users_info = pd.DataFrame(users_info, columns=['coluna', 'tipo', 'nulos', 'unicos'])
print(users_info)

# Ações do tipo 'play'
df_play = users[users['action'] == 'play']
print("Amostra de ações do tipo 'play': \n")
print(df_play.sample(5))

# Top 10 jogos mais jogados
top_played = df_play.groupby('game')['value'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
top_played.plot(kind='barh', color='skyblue')
plt.xlabel("Tempo total jogado")
plt.title("Top 10 jogos mais jogados (steam-200k.csv)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Quantidade média de jogos por usuário
jogos_por_usuario = df_play.groupby('user_id')['game'].count()
plt.figure(figsize=(8,5))
sns.histplot(jogos_por_usuario, bins=50, kde=True)
plt.title("Distribuição de jogos jogados por usuário")
plt.xlabel("Número de jogos")
plt.grid(True)
plt.show()

# Distribuição do tempo de jogo
plt.figure(figsize=(8,5))
sns.histplot(df_play['value'], bins=100, kde=True)
plt.title("Distribuição do tempo jogado (em minutos)")
plt.xlabel("Minutos")
plt.grid(True)
plt.show()