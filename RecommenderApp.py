import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset
movies = pd.read_csv("movie.csv")  # movie dataset
ratings = pd.read_csv("ratings_small.csv")  # ratings dataset

# Merge movies and ratings
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix for collaborative filtering
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Fit a Nearest Neighbors model for collaborative filtering
knn_model_user = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model_user.fit(user_movie_ratings)

# Fit a TfidfVectorizer for content-based filtering based on movie genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'].fillna(''))

# Fit a Nearest Neighbors model for content-based filtering
knn_model_content = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model_content.fit(tfidf_matrix)

# Streamlit App
st.header("Personalized Movie Recommendation Engine",divider="rainbow")

# User input: userId and movie title
user_id = st.number_input("Enter your userId", min_value=1, max_value=user_movie_ratings.index.max(), value=1)

# User input for movie title with autocomplete options
movie_title = st.selectbox("Select a movie title", movies['title'].tolist())

# Get the movieId for the entered movie title
movie_id_user = movies.loc[movies['title'] == movie_title, 'movieId'].values

if len(movie_id_user) == 0:
    st.warning("Movie not found. Please enter a valid movie title.")
else:
    movie_id_user = movie_id_user[0]

    # Get the top 10 recommended movies based on user ratings
    distances_user, indices_user = knn_model_user.kneighbors(user_movie_ratings.loc[user_id].values.reshape(1, -1), n_neighbors=11)
    recommended_movies_user = [movies.iloc[idx]['title'] for idx in indices_user.flatten()[1:]]

    # Get the top 10 recommended movies based on content similarity
    movie_index_content = movies[movies['movieId'] == movie_id_user].index[0]
    distances_content, indices_content = knn_model_content.kneighbors(tfidf_matrix[movie_index_content], n_neighbors=11)
    recommended_movies_content = [movies.iloc[idx]['title'] for idx in indices_content.flatten()[1:]]

    
    combined_recommendations = list(set(recommended_movies_user) | set(recommended_movies_content))
    # Remove the selected movie from recommendations
    combined_recommendations = [movie for movie in combined_recommendations if movie != movie_title]

    # Display the filtered recommendations
    st.subheader("Recommended Movies:")
    for movie in combined_recommendations[:12]:
        st.write(movie)

st.divider()
# Allow users to provide feedback
user_feedback = st.selectbox("How would you rate the recommendations?", [5, 4, 3, 2, 1])

# You can save the feedback in your dataset for future model improvement
st.write(f"Thank you for your feedback! You rated the recommendations as: {user_feedback}")
