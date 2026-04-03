import gradio as gr
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from recommender import recommend


load_dotenv()


movies = pd.read_csv("data/movies.csv")
API_KEY = os.getenv("API_KEY")

def clean_movie_name(title):
    title = title.split("(")[0].strip()

    if "," in title:
        parts = title.split(",")
        title = parts[1].strip() + " " + parts[0].strip()

    return title


def fetch_poster(movie_name):
    try:
        movie_name = movie_name.replace(" ", "+")
        url = f"http://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"
        data = requests.get(url).json()

        if data.get("Response") == "True" and data.get("Poster") != "N/A":
            return data["Poster"]
        else:
            return "https://via.placeholder.com/300x450?text=No+Poster"

    except:
        return "https://via.placeholder.com/300x450?text=Error"


def interface(movie_name):
    try:
        if not movie_name.strip():
            return "Please enter movie", None, []

        matched = movies[movies["title"].str.contains(movie_name, case=False, na=False)]

        if matched.empty:
            return "Movie not found", None, []

        movie_index = matched.index[0]
        movie_title = matched.iloc[0]["title"]

        # selected poster
        selected_poster = fetch_poster(clean_movie_name(movie_title))

        # recommendations
        names = recommend(movie_index)
        selected_genre = movies.iloc[movie_index]['genres']
        filtered = movies[movies['genres'].str.contains(selected_genre.split('|')[0], na=False)]
        names = filtered['title'].head(5).tolist()

        posters = []
        for movie in names:
            poster = fetch_poster(clean_movie_name(movie))

           
            if not poster:
                poster = "https://via.placeholder.com/300x450?text=No+Poster"

            posters.append((poster, movie))

        return movie_title, selected_poster, posters

    except Exception as e:
        return f"Error: {str(e)}", None, []


demo = gr.Interface(
    fn=interface,
    inputs=gr.Textbox(label="Enter Movie Name"),
    outputs=[
        gr.Textbox(label="Selected Movie"),
        gr.Image(label="Selected Movie Poster"),
        gr.Gallery(label="Recommendations")
    ],
    title=" Netflix AI Recommendation System",
    description="Enter a movie name and get 5 similar movies with posters"
)

demo.launch()