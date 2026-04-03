import numpy as np
import faiss
import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

embeddings = np.load(os.path.join(BASE_DIR, "embeddings", "movie_embeddings.npy"))
movies = pd.read_csv(os.path.join(BASE_DIR, "data", "movies.csv"))

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def recommend(movie_id, top_k=5):
    query = embeddings[movie_id].reshape(1,-1)
    distances, indices = index.search(query, top_k)
    return movies.iloc[indices[0]]["title"].tolist()