import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

film_list = pickle.load(open('filmler.pkl', 'rb'))
filmler = pd.DataFrame(film_list)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(filmler["tags"])

def search(tags):
    query = vectorizer.transform([tags])
    similarity_scores = cosine_similarity(query, tfidf).flatten()
    indices = np.argpartition(similarity_scores, -5)[-5:]
    results = filmler.iloc[indices][::-1]
    return results

def fetch_poster_url(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0ZTRlYWFiMTI5NGQ2NTY3M2JiYmE5MThiZTBiOTE2OSIsInN1YiI6IjY2NGUyMmEzNWNjOGNlMTk0ZTQzY2ExNCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.3NCu-si_HkDRlTmcAFA3DZfoWbjm_-iTQCingt_6aPc"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        posters = data.get('posters')
        if posters:
            poster_path = posters[0]['file_path']
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        print(f"Error fetching data for movie ID {movie_id}: {response.status_code}")
        print(response.json())
    return None


st.title("Film Öneri Sistemi")

film_name = st.text_input("Film adı girin:")

if st.button('Öner'):
    if film_name:
        recommendations = search(film_name)
        st.write("Önerilen Filmler:")
        for index, row in recommendations.iterrows():
            st.write(f"Film: {row['title']}")
            movie_id = row['movie_id']
            st.write(f"Movie ID: {movie_id}")  # Debugging step
            poster_url = fetch_poster_url(movie_id)
            if poster_url:
                st.image(poster_url, width=200)  # Adjust poster size here
            else:
                st.write("Poster bulunamadı.")
    else:
        st.write("Lütfen bir film adı girin.")
