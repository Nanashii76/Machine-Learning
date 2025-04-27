import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- CARREGAR E PREPARAR OS DADOS ---

@st.cache_data
def carregar_dados():
    steam = pd.read_csv('./data/steam.csv')
    users = pd.read_csv('./data/steam-200k.csv')
    users.columns = ['user_id', 'game', 'action', 'value', 'unused']

    # Preprocessamento
    steam['owners_avg'] = steam['owners'].apply(lambda x: np.mean([int(i.replace(',', '')) for i in x.split('-')]) if pd.notnull(x) else 0)
    steam['release_date'] = pd.to_datetime(steam['release_date'], errors='coerce')
    steam['release_year'] = steam['release_date'].dt.year
    steam['name'] = steam['name'].str.lower()
    users['game'] = users['game'].str.lower()
    
    df_play = users[users['action'] == 'play']
    merged = pd.merge(df_play, steam, left_on='game', right_on='name', how='inner')
    merged['genres'] = merged['genres'].str.split(';')
    merged = merged.explode('genres')
    merged['value'] = pd.to_numeric(merged['value'], errors='coerce').fillna(0).astype(int)
    merged = merged[merged['value'] < 10000]
    
    dados = merged[['user_id', 'game', 'value', 'positive_ratings', 'negative_ratings',
                    'average_playtime', 'price', 'release_year', 'owners_avg', 'genres']].copy()
    
    dados['genre_list'] = dados['genres']
    scaler = MinMaxScaler()
    dados[['positive_ratings_norm', 'negative_ratings_norm',
           'average_playtime_norm', 'price_norm', 'owners_avg_norm']] = scaler.fit_transform(
        dados[['positive_ratings', 'negative_ratings', 'average_playtime', 'price', 'owners_avg']]
    )
    dados = dados.drop(columns=['positive_ratings', 'negative_ratings',
                                'average_playtime', 'price', 'owners_avg', 'genres'])

    return dados

dados = carregar_dados()

# --- CONSTRUIR MODELO DE RECOMENDAÇÃO ---

@st.cache_data
def construir_recomendador(dados):
    jogos = dados.groupby('game').agg({
        'positive_ratings_norm': 'mean',
        'negative_ratings_norm': 'mean',
        'average_playtime_norm': 'mean',
        'price_norm': 'mean',
        'owners_avg_norm': 'mean',
        'genre_list': lambda x: list(x)[0]
    }).reset_index()

    generos_dummies = pd.get_dummies(jogos['genre_list'], prefix='genre')
    jogos_features = pd.concat([jogos[['game', 'positive_ratings_norm', 'negative_ratings_norm',
                                       'average_playtime_norm', 'price_norm', 'owners_avg_norm']],
                                generos_dummies], axis=1)
    
    features = jogos_features.drop(columns=['game'])
    similaridade = cosine_similarity(features)

    return jogos_features, similaridade

jogos_features, similaridade = construir_recomendador(dados)

# --- FUNÇÃO DE RECOMENDAÇÃO ---

def recomendar_jogos(nome_jogo, num_recomendacoes=5):
    nome_jogo = nome_jogo.lower()
    if nome_jogo not in jogos_features['game'].values:
        return None
    idx = jogos_features[jogos_features['game'] == nome_jogo].index[0]
    
    similaridades = list(enumerate(similaridade[idx]))
    similaridades = sorted(similaridades, key=lambda x: x[1], reverse=True)[1:num_recomendacoes+1]
    indices = [i[0] for i in similaridades]
    return jogos_features.iloc[indices][['game', 'positive_ratings_norm', 'average_playtime_norm', 'price_norm']]