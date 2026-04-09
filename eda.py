import pandas as pd
import matplotlib.pyplot as plt

print("\n==== EDA START ====\n")
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

# Missing values
print("\nMissing values (movies):\n", movies.isnull().sum())
print("\nMissing values (ratings):\n", ratings.isnull().sum())

# Unique users & movies
print("Unique users:", ratings['userId'].nunique())
print("Unique movies:", ratings['movieId'].nunique())
# Rating distribution
ratings['rating'].hist(bins=20)
plt.title("Rating Distribution")
plt.show()

# Top movies
top_movies = ratings.groupby('movieId')['rating'].count().sort_values(ascending=False).head(10)
print("Top Movies:\n", top_movies)

top_movies.plot(kind='bar')
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Movie ID")
plt.ylabel("Number of Ratings")
plt.show()

# Average rating per movie
avg_rating = ratings.groupby('movieId')['rating'].mean()
print("Average rating:\n", avg_rating.head())

# Most active users
active_users = ratings.groupby('userId')['rating'].count().sort_values(ascending=False).head(10)
print("Most active users:\n", active_users)

# Genre analysis
movies['genres'] = movies['genres'].str.split('|')

from collections import Counter

genre_list = [genre for sublist in movies['genres'] for genre in sublist]
genre_count = Counter(genre_list)

print("Top genres:\n", genre_count.most_common(10))

genre_df = pd.DataFrame(genre_count.most_common(10), columns=['Genre', 'Count'])
genre_df.plot(kind='bar', x='Genre', y='Count')
plt.title("Top Genres")
plt.show()

print("\n==== EDA COMPLETE ====")