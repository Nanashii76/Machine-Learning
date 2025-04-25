import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Carregar as bases de dados Steam
steam = pd.read_csv('./data/steam.csv')
users = pd.read_csv('./data/steam-200k.csv')
users.columns = ['user_id', 'game', 'action', 'value', 'unused']

# Criar coluna owners_avg (média da faixa de donos)
steam['owners_avg'] = steam['owners'].apply(
    lambda x: np.mean([int(i.replace(',', '')) for i in x.split('-')]) if pd.notnull(x) else 0
)

# Criar coluna release_year a partir de release_date
steam['release_date'] = pd.to_datetime(steam['release_date'], errors='coerce')
steam['release_year'] = steam['release_date'].dt.year

'''
    Análise steam.csv (jogos)
'''
print("Análise de atributos do dataset steam.csv\n")
jogos_info = []
for col in steam.columns:
    jogos_info.append([col, steam[col].dtype, steam[col].isnull().sum(), steam[col].nunique()])
jogos_info = pd.DataFrame(jogos_info, columns=['coluna', 'tipo', 'nulos', 'unicos'])
print(jogos_info)

# Correlação entre variáveis numéricas
plt.figure(figsize=(10, 8))
sns.heatmap(steam[['positive_ratings', 'negative_ratings', 'average_playtime', 'price']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlação entre variáveis numéricas')
plt.show()

'''
    Análise steam-200k.csv (usuários)
'''
print("Análise de atributos do dataset steam-200k.csv\n")
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

'''
    Junção dos dados
'''

# Padronizar nomes dos jogos para minúsculas
steam['name'] = steam['name'].str.lower()
df_play['game'] = df_play['game'].str.lower()

# Merge entre users e steam
merged_df = pd.merge(df_play, steam, left_on='game', right_on='name', how='inner')

# Comparações
print("\nTotal de registros em steam-200k.csv:", users.shape[0])
print("Total de registros em steam.csv:", steam.shape[0])
print("Total de registros após o merge:", merged_df.shape[0])

'''
    Análise do merged_df
'''

print("\nAnálise de atributos do dataset merged_df\n")
merged_info = []
for col in merged_df.columns:
    merged_info.append([col, merged_df[col].dtype, merged_df[col].isnull().sum(), merged_df[col].nunique()])
merged_info = pd.DataFrame(merged_info, columns=['coluna', 'tipo', 'nulos', 'unicos'])
print(merged_info)

# Explodir múltiplos gêneros separados por ;
merged_df['genres'] = merged_df['genres'].str.split(';')
merged_df = merged_df.explode('genres')

# Tratar coluna 'value'
merged_df['value'] = pd.to_numeric(merged_df['value'], errors='coerce').fillna(0).astype(int)

# Remover outliers extremos para visualização
merged_df = merged_df[merged_df['value'] < 10000]

# Tempo total jogado por gênero
plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='value', y='genres', estimator=sum, ci=None, palette='viridis')
plt.title('Tempo total jogado por gênero')
plt.xlabel('Tempo total jogado (em minutos)')
plt.ylabel('Gênero')
plt.grid(True)
plt.show()

'''
    Processamento final para análise/modelo
'''

# Selecionar colunas úteis
colunas_selecionadas = [
    'user_id', 'game', 'value', 
    'positive_ratings', 'negative_ratings', 'average_playtime', 
    'price', 'release_year', 'owners_avg', 'genres'
]
dados = merged_df[colunas_selecionadas].copy()

# Copiar gêneros já tratados
dados['genre_list'] = dados['genres']

# Normalizar colunas numéricas
scaler = MinMaxScaler()
dados[['positive_ratings_norm', 'negative_ratings_norm', 
       'average_playtime_norm', 'price_norm', 'owners_avg_norm']] = scaler.fit_transform(
    dados[['positive_ratings', 'negative_ratings', 'average_playtime', 'price', 'owners_avg']]
)

# Remover colunas brutas para limpeza
dados = dados.drop(columns=['positive_ratings', 'negative_ratings', 'average_playtime', 'price', 'owners_avg', 'genres'])

# Visualizar resultado final
print("\nDataset final pronto para recomendação:\n")
print(dados.head())


'''
 Construindo sistema de recomendação
'''

# Agrupar para termos um vetor único por jogo
jogos = dados.groupby('game').agg({
    'positive_ratings_norm': 'mean',
    'negative_ratings_norm': 'mean',
    'average_playtime_norm': 'mean',
    'price_norm': 'mean',
    'owners_avg_norm': 'mean',
    'genre_list': lambda x: list(x)[0]  # Pega o primeiro gênero (já está explodido)
}).reset_index()

# One-hot encoding dos gêneros
generos_dummies = pd.get_dummies(jogos['genre_list'], prefix='genre')

# concatenar os gêneros dummies com o dataframe original
jogos_features = pd.concat([jogos[['game', 'positive_ratings_norm', 'negative_ratings_norm', 
                                   'average_playtime_norm', 'price_norm', 'owners_avg_norm']], 
                            generos_dummies], axis=1)

# Separar só as features
features = jogos_features.drop(columns=['game'])

# Calcular matriz de similaridade
similaridade = cosine_similarity(features)

# Criar função para recomendar jogos
def recomendar_jogos(nome_jogo, num_recomendacoes=5):
    # Verificar se o jogo existe
    if nome_jogo not in jogos_features['game'].values:
        print(f"Jogo '{nome_jogo}' não encontrado.")
        return
    
    # Obter o índice do jogo
    idx = jogos_features[jogos_features['game'] == nome_jogo].index[0]
    
    # Obter as similaridades e os índices dos jogos mais similares
    similaridades = list(enumerate(similaridade[idx]))
    similaridades = sorted(similaridades, key=lambda x: x[1], reverse=True)[1:num_recomendacoes+1]
    
    # Obter os índices dos jogos recomendados
    indices_recomendados = [i[0] for i in similaridades]
    
    # Retornar os jogos recomendados
    return jogos_features.iloc[indices_recomendados][['game', 'positive_ratings_norm', 'negative_ratings_norm', 
                                                       'average_playtime_norm', 'price_norm', 'owners_avg_norm']]

# teste da função de recomendação
# print("\nRecomendações para o jogo 'Counter-Strike':\n")
# recomendacoes = recomendar_jogos('counter-strike', num_recomendacoes=5)
# print(recomendacoes)
