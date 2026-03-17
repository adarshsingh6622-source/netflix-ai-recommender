import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("../model/netflix_model.h5", compile=False)

movie_embeddings = model.layers[3].get_weights()[0]

np.save("movie_embeddings.npy", movie_embeddings)