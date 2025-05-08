import pandas as pd

movies = pd.read_csv("./datas/movies.csv")
ratings = pd.read_csv("./datas/ratings.csv")
tags = pd.read_csv("./datas/tags.csv")
links = pd.read_csv("./datas/links.csv")

ratings.columns = ["usuarioId", "filmeId", "nota", "momento"]
print(ratings["nota"].value_counts())
print(ratings["nota"].mean())

ratings["nota"].plot(kind="hist")