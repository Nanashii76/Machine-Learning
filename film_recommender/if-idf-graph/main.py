import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import math 

movies_data = {
    'movieId': [1, 2, 3, 4, 5, 6],
    'title': [
        'Jurassic Park (1993)',
        'Forrest Gump (1994)',
        'Toy Story (1995)',
        'Die Hard (1988)',
        'Pulp Fiction (1994)',
        'Up (2009)'
    ],
    'genres': [
        'Adventure|Sci-Fi|Thriller',
        'Comedy|Drama|Romance',
        'Animation|Children|Comedy',
        'Action|Thriller',
        'Crime|Drama',
        'Animation|Adventure|Comedy'
    ]
}
movies_df = pd.DataFrame(movies_data)

movies_df['genres_processed'] = movies_df['genres'].apply(lambda x: x.replace('|', ' '))

vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
tfidf_matrix = vectorizer.fit_transform(movies_df['genres_processed'])

feature_names = vectorizer.get_feature_names_out()

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(tfidf_matrix.toarray())

movies_df['pca_x'] = reduced_vectors[:, 0]
movies_df['pca_y'] = reduced_vectors[:, 1]


plt.figure(figsize=(10, 10))
ax = plt.gca()

colors = plt.cm.get_cmap('tab10', len(movies_df))

for i, row in movies_df.iterrows():
    ax.arrow(0, 0, row['pca_x'], row['pca_y'],
             head_width=0.05, head_length=0.07,
             fc=colors(i), ec=colors(i),
             label=row['title']) # Cor para cada filme
    ax.text(row['pca_x'] * 1.05, row['pca_y'] * 1.05, row['title'],
            color=colors(i), fontsize=9, weight='bold') # Nome do filme

print("\n--- Ângulos de Similaridade de Cosseno entre Filmes (em Graus) ---")

movie1_idx = movies_df[movies_df['title'] == 'Jurassic Park (1993)'].index[0]
movie2_idx = movies_df[movies_df['title'] == 'Die Hard (1988)'].index[0]

vec1 = tfidf_matrix[movie1_idx].toarray().flatten()
vec2 = tfidf_matrix[movie2_idx].toarray().flatten()

cosine_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
angle_rad = np.arccos(np.clip(cosine_sim, -1.0, 1.0)) # np.clip para evitar erros de ponto flutuante fora de [-1, 1]
angle_deg = math.degrees(angle_rad)
print(f"'{movies_df.loc[movie1_idx, 'title']}' e '{movies_df.loc[movie2_idx, 'title']}':")
print(f"  Similaridade de Cosseno: {cosine_sim:.4f}, Ângulo: {angle_deg:.2f}°")

movie3_idx = movies_df[movies_df['title'] == 'Forrest Gump (1994)'].index[0]
movie4_idx = movies_df[movies_df['title'] == 'Pulp Fiction (1994)'].index[0]

vec3 = tfidf_matrix[movie3_idx].toarray().flatten()
vec4 = tfidf_matrix[movie4_idx].toarray().flatten()

cosine_sim_diff = cosine_similarity(vec3.reshape(1, -1), vec4.reshape(1, -1))[0][0]
angle_rad_diff = np.arccos(np.clip(cosine_sim_diff, -1.0, 1.0))
angle_deg_diff = math.degrees(angle_rad_diff)
print(f"'{movies_df.loc[movie3_idx, 'title']}' e '{movies_df.loc[movie4_idx, 'title']}':")
print(f"  Similaridade de Cosseno: {cosine_sim_diff:.4f}, Ângulo: {angle_deg_diff:.2f}°")

movie5_idx = movies_df[movies_df['title'] == 'Toy Story (1995)'].index[0]
movie6_idx = movies_df[movies_df['title'] == 'Up (2009)'].index[0]

vec5 = tfidf_matrix[movie5_idx].toarray().flatten()
vec6 = tfidf_matrix[movie6_idx].toarray().flatten()

cosine_sim_anim = cosine_similarity(vec5.reshape(1, -1), vec6.reshape(1, -1))[0][0]
angle_rad_anim = np.arccos(np.clip(cosine_sim_anim, -1.0, 1.0))
angle_deg_anim = math.degrees(angle_rad_anim)
print(f"'{movies_df.loc[movie5_idx, 'title']}' e '{movies_df.loc[movie6_idx, 'title']}':")
print(f"  Similaridade de Cosseno: {cosine_sim_anim:.4f}, Ângulo: {angle_deg_anim:.2f}°")

ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title("Visualização de Filmes no Espaço de Gêneros (PCA 2D)")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")

plt.show()