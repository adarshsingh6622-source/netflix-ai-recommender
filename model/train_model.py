import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("../data/ratings.csv")

user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_map = {x:i for i,x in enumerate(user_ids)}
movie_map = {x:i for i,x in enumerate(movie_ids)}

ratings["user"] = ratings["userId"].map(user_map)
ratings["movie"] = ratings["movieId"].map(movie_map)

num_users = len(user_map)
num_movies = len(movie_map)

X = ratings[["user","movie"]].values
y = ratings["rating"].values

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)

embedding_size = 50

user_input = tf.keras.layers.Input(shape=(1,))
user_embed = tf.keras.layers.Embedding(num_users,embedding_size)(user_input)
user_vec = tf.keras.layers.Flatten()(user_embed)

movie_input = tf.keras.layers.Input(shape=(1,))
movie_embed = tf.keras.layers.Embedding(num_movies,embedding_size)(movie_input)
movie_vec = tf.keras.layers.Flatten()(movie_embed)

concat = tf.keras.layers.Concatenate()([user_vec,movie_vec])
dense = tf.keras.layers.Dense(128,activation="relu")(concat)
output = tf.keras.layers.Dense(1)(dense)

model = tf.keras.Model([user_input,movie_input],output)

model.compile(loss="mse",optimizer="adam")

model.fit(
    [X_train[:,0],X_train[:,1]],
    y_train,
    validation_data=([X_val[:,0],X_val[:,1]],y_val),
    epochs=5
)

model.save("netflix_model.h5")