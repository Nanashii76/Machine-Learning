import streamlit as st
import profile_rec as prf
import func as fc


dados = fc.carregar_dados()
jogos_features, similaridade = fc.construir_recomendador(dados)
# --- INTERFACE COM STREAMLIT ---

st.title("🎮 Recomendador de Jogos Steam")

abas = st.tabs(["Buscar por nome do jogo", "Recomendações baseadas no seu perfil Steam"])

with abas[0]:
    lista_jogos = sorted(jogos_features['game'].unique())
    termo_busca = st.text_input("Digite o nome de um jogo que você goste: ", "")

    jogos_filtrados = [jogo for jogo in lista_jogos if termo_busca.lower() in jogo.lower()]

    # -- FACILITAR A BUSCA -- 
    if len(jogos_filtrados) > 100 and termo_busca:
        st.warning(f"Encontrados {len(jogos_filtrados)} jogos. Mostrando apenas os primeiros 100. Digite mais letras para refinar a busca.")
        jogos_filtrados = jogos_filtrados[:100]

    if termo_busca and jogos_filtrados:
        jogo_selecionado = st.selectbox(
            "Selecione um jogo da lista:",
            options=jogos_filtrados,
            index=0
        )
        
        if st.button("Recomendar"):
            recomendacoes = fc.recomendar_jogos(jogo_selecionado, num_recomendacoes=5)
            if recomendacoes is not None:
                st.success(f"Jogos similares a '{jogo_selecionado}':")
                
                df_display = recomendacoes.rename(columns={
                    'game': 'Jogo',
                    'positive_ratings_norm': 'Avaliação',
                    'average_playtime_norm': 'Tempo médio',
                    'price_norm': 'Preço'
                })
                
                st.dataframe(df_display)
            else:
                st.error("Erro ao gerar recomendações. Tente outro jogo.")
    elif termo_busca:
        st.warning("Nenhum jogo encontrado com esse termo. Tente outra palavra.")
with abas[1]:
    prf.app()