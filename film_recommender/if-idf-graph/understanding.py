from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Arroz e feijão faz bem para a saúde",
    "Feijão é muito bom",
    "Dieta com arroz e feijão",
    "O homem é condenado a ser livre"
]

# stop_words='portuguese' usa a lista padrão de stop words do sklearn para português
# ou você pode passar a lista do NLTK como fizemos antes
vectorizer = TfidfVectorizer(lowercase=True)
tfidf_matrix = vectorizer.fit_transform(documents)

print("Formato da matriz TF-IDF:", tfidf_matrix.shape)
print("Nomes dos termos (vocabulário):", vectorizer.get_feature_names_out())
print("Matriz TF-IDF (primeiras linhas):\n", tfidf_matrix.toarray().round(4))