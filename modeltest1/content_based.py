import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv('movies.csv', encoding='latin_1', sep='\t', usecols=['movie_id', 'title', 'genres'])
movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' ').replace('-', ''))
# movies.head()
# movies.info()
# movies.describe()

users = pd.read_csv("users.csv", encoding='latin_1', sep="\t")
ratings = pd.read_csv("ratings.csv", encoding='latin_1', sep="\t")

plt.figure()
sns.countplot(x=ratings['rating'])
plt.show()
title = "Jumanji (1995)"

num_recommendations = 20

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(movies['genres'])
# print(matrix.shape)
# print(vectorizer.vocabulary_)
# print(matrix.todense())
similarity_matrix = cosine_similarity(matrix)
similarity_data = pd.DataFrame(similarity_matrix, index=movies['title'], columns=movies['title'])

result = similarity_data.loc[title, :].sort_values(ascending=False)[:num_recommendations].drop(title).to_frame(name="score").reset_index()

result = result.merge(movies[['genres','title']]) # lay them thong tin genres cho ten phim
print(result)