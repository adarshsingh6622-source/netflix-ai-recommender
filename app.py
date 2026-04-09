import streamlit as st
import pandas as pd
from recommender import recommend
import os
import requests
from dotenv import load_dotenv
load_dotenv()
import logging

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#  API KEY
API_KEY = os.getenv("API_KEY")

#  Path fix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies = pd.read_csv(os.path.join(BASE_DIR, "data", "movies.csv"))

#  Movie name clean function
def clean_movie_name(title):
    title = title.split('(')[0].strip()

    if ',' in title:
        parts = title.split(',')
        title = parts[1].strip() + " " + parts[0].strip()

    return title

#  Poster fetch function
def fetch_poster(movie_name):
    try:
        url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={movie_name}"
        data = requests.get(url).json()

        if data["Response"] == "True":
            return data["Poster"]
        return None
    except:
        return None

#  UI
st.set_page_config(page_title="Netflix AI", layout="centered")

st.title(" Netflix AI Recommendation System")
st.write("Enter a movie name and get similar recommendations ")

#  Input
movie_name = st.text_input(" Enter Movie Name")

#  Button
if st.button("Recommend"):
    logger.info(f"User clicked recommend button with input: {movie_name}")

    if movie_name.strip() == "":
        st.warning("⚠️ Please enter a movie name")

    else:
        #  Search movie
        matched = movies[movies["title"].str.contains(movie_name, case=False, na=False)]
        logger.info(f"Movies matched: {len(matched)}")

        if matched.empty:
            st.error(" Movie not found")
            logger.warning(f"Movie not found: {movie_name}")

        else:
            #  Pick first match
            movie_id = int(matched.index[0])
            movie_title = matched.iloc[0]["title"]

            st.success(f" Selected Movie: {movie_title}")

            #  Selected movie poster
            clean_title = clean_movie_name(movie_title)
            poster = fetch_poster(clean_title)

            if poster and poster != "N/A":
                st.image(poster, width=200)

            st.markdown(f"###  {movie_title}")

            #  Recommendations
            results = recommend(movie_id)
            logger.info(f"Recommendations generated for movie_id {movie_id}")

            st.subheader(" Recommended Movies:")

            #  Show recommended movies
            for movie in results:
                clean_name = clean_movie_name(movie)
                poster = fetch_poster(clean_name)

                if poster and poster != "N/A":
                    st.image(poster, width=150)

                st.write(movie)
