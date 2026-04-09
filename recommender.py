import numpy as np
import faiss
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "movie_embeddings.npy")
MOVIES_PATH = os.path.join(BASE_DIR, "data", "movies.csv")

try:
    embeddings = np.load(EMBEDDINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    print("Data loaded successfully")
except Exception as e:
    print("Error loading data", e)
    raise

try:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("FAISS index created")
except Exception as e:
    print("Error creating FAISS index:", e)
    raise
    

# Cold Start Function


def cold_start(top_k=5):
    try:
        return movies['title'].value_counts().head(top_k).index.tolist()
    except Exception as e:
        print("Error in cold_start:", e)
        return ["No data available"]

    

def recommend(movie_id, top_k=5):
    if not isinstance(movie_id, int):
        return ["movie_id must be integer"]
    if movie_id < 0:
        return ["Invalid movie id"]
    try:
        print(f"Recommending for movie_id: {movie_id}")
        if movie_id >= len(embeddings):
             print("Using cold_start")

             return cold_start(top_k)

       
        query = embeddings[movie_id].reshape(1, -1)
        distances, indices = index.search(query, top_k+1)
        rec_ids = indices[0][1:top_k+1]
        recommended = movies.iloc[rec_ids]['title'].tolist()
        recommended = list(set(recommended))
        return sorted(recommended)
    except Exception as e:
        print("Error in recommendation:", e)
        return ["Error occured"]
    
from sklearn.metrics.pairwise import cosine_similarity

def cosine_recommend(movie_id, top_k=5):
    if not isinstance(movie_id, int):
        return ["movie_id must be integer"]
    try:
        print(f"Cosine recommendation for movie_id: {movie_id}")
        sim = cosine_similarity(embeddings)
        scores = sim[movie_id]

        top_indices = scores.argsort()[::-1][1:top_k+1]

        return movies.iloc[top_indices]['title'].tolist()
    except Exception as e:
        print("Error in cosine recommendation:", e)
        return ["Error occured"]
    

        
    