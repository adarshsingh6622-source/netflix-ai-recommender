import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(level=logging.INFO)

logging.info("Training started...")

ratings = pd.read_csv("data/ratings.csv")

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

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

embedding_size = 64

user_input = tf.keras.layers.Input(shape=(1,))
user_embed = tf.keras.layers.Embedding(num_users,embedding_size)(user_input)
user_vec = tf.keras.layers.Flatten()(user_embed)

movie_input = tf.keras.layers.Input(shape=(1,))
movie_embed = tf.keras.layers.Embedding(num_movies,embedding_size)(movie_input)
movie_vec = tf.keras.layers.Flatten()(movie_embed)

concat = tf.keras.layers.Concatenate()([user_vec,movie_vec])
dense = tf.keras.layers.Dense(128,activation="relu")(concat)
drop = tf.keras.layers.Dropout(0.3)(dense)
dense2 = tf.keras.layers.Dense(64,activation='relu')(drop)
output = tf.keras.layers.Dense(1)(dense2)

model = tf.keras.Model([user_input,movie_input],output)

model.compile(loss="mse",optimizer="adam",metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)

history = model.fit(
    [X_train[:,0],X_train[:,1]],
    y_train,
    validation_data=([X_val[:,0],X_val[:,1]],y_val),
    epochs=10,
    batch_size=64,
    callbacks=[early_stop]
)

loss, mae = model.evaluate([X_val[:,0], X_val[:,1]], y_val)
print("Final MAE:", mae)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

model.save("netflix_model_v1.h5")