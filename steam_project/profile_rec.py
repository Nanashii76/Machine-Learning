import streamlit as st
import requests
import pandas as pd
import numpy as np
import func as fc

URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
API_KEY = ""

def get_player_games(id_steam):
  params = {
    "key": API_KEY,
    "steamid": id_steam,
    "include_appinfo": True,
    "include_played_free_games": True
  }

  response = requests.get(URL, params=params)
  if response.status_code != 200:
    return -1  

  data = response.json()
  if "response" not in data or "games" not in data["response"]:
    return -1
  
  games = data["response"]["games"]

  df = pd.DataFrame(games)
  if not df.empty:
    df['playtime_hours'] = df['playtime_forever'] / 60
    df['name'] = df['name'].str.lower()
    df = df[df['playtime_forever'] > 30] 

  return df

def app():
  with st.expander("Como encontrar meu ID Steam?"):
    st.write("""
    1. Acesse [sua conta Steam](https://store.steampowered.com/account/)
    2. O número do seu ID Steam aparece na seção de informações da conta
    3. Também pode ser encontrado no URL do seu perfil: steamcommunity.com/profiles/[seu_id_steam]
    
    Atenção: Certifique-se de que seu perfil esteja público para que possamos acessar sua lista de jogos!
    """)

  id_steam = st.text_input("ID Steam", "")

  if st.button("Buscar meus jogos e recomendar", type="primary", key="btn_rec_id"):
    if not id_steam:
      st.warning("Por favor, digite um ID Steam válido.")
    else:
      with st.spinner("Conectando à Steam e processando seus jogos..."):
        jogos_do_usuario = get_player_games(id_steam) 

        if jogos_do_usuario == -1:
          st.error("Erro ao conectar à Steam ou ID Steam inválido. Tente novamente.")
        else:
          if jogos_do_usuario is not None and not jogos_do_usuario.empty:
            st.success(f"Encontrados {len(jogos_do_usuario)} jogos em sua biblioteca!")
            with st.expander("Ver seus jogos analisados"):
              st.dataframe(
                jogos_do_usuario[['name', 'playtime_hours']].sort_values('playtime_hours', ascending=False),
                column_config={
                  "name": "Nome do jogo",
                  "playtime_hours": st.column_config.NumberColumn("Tempo de jogo (horas)", format="%.1f h")
                },
                use_container_width=True
              )
            dados = fc.carregar_dados()
            jogos_features, similaridade = fc.construir_recomendador(dados)
            jogos_validos = jogos_do_usuario[jogos_do_usuario['name'].isin(jogos_features['game'])]

            if jogos_validos.empty:
              st.warning("Nenhum dos seus jogos foi encontrado na base de dados.")
            else:
              top_jogos = jogos_validos.sort_values('playtime_hours', ascending=False).head(5)['name'].tolist()

              st.info(f"Gerando recomendações baseadas nos jogos: {' | '.join(top_jogos)}")

              indices = [jogos_features[jogos_features['game'] == jogo].index[0] for jogo in top_jogos]

              media_similaridade = np.mean(similaridade[indices], axis=0)

              recomendados_idx = np.argsort(media_similaridade)[::-1]
              recomendados_idx = [i for i in recomendados_idx if jogos_features.iloc[i]['game'] not in top_jogos][:5]

              recomendados = jogos_features.iloc[recomendados_idx][['game', 'positive_ratings_norm', 'average_playtime_norm', 'price_norm']]

              st.success("Jogos recomendados para você:")

              df_display = recomendados.rename(columns={
                  'game': 'Jogo',
                  'positive_ratings_norm': 'Avaliação',
                  'average_playtime_norm': 'Tempo médio',
                  'price_norm': 'Preço'
              })

              st.dataframe(df_display)


