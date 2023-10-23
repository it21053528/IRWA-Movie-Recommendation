import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset
movies = pd.read_csv("movie.csv")  # Replace with your actual movie dataset
ratings = pd.read_csv("ratings_small.csv")  # Replace with your actual ratings dataset

# Merge movies and ratings
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Collaborative filtering

# Create a user-item matrix for collaborative filtering
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Fit a Nearest Neighbors model for collaborative filtering
knn_model_user = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model_user.fit(user_movie_ratings)

# Content-based filtering

# Combine title and genre for content-based filtering
movies['title_and_genre'] =  movies['genres']  + ' ' + movies['title']

# Fit a TfidfVectorizer for content-based filtering based on movie title and genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_combined = tfidf_vectorizer.fit_transform(movies['title_and_genre'].fillna(''))

# Fit a Nearest Neighbors model for content-based filtering
knn_model_content = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model_content.fit(tfidf_matrix_combined)

# Streamlit App
st.header("Personalized Movie Recommendation Engine", divider="rainbow")

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

    # Collaborative Filtering Recommendations
    distances_user, indices_user = knn_model_user.kneighbors(user_movie_ratings.loc[user_id].values.reshape(1, -1), n_neighbors=11)
    recommended_movies_user = [(movies.iloc[idx]['title'], 1 - distances_user.flatten()[i]) for i, idx in enumerate(indices_user.flatten()[1:])]

    # Content-Based Filtering Recommendations
    movie_index_content = movies[movies['movieId'] == movie_id_user].index[0]
    distances_content, indices_content = knn_model_content.kneighbors(tfidf_matrix_combined[movie_index_content], n_neighbors=11)
    recommended_movies_content = [(movies.iloc[idx]['title'], 1 - distances_content.flatten()[i]) for i, idx in enumerate(indices_content.flatten()[1:])]

    # Combine Recommendations
    combined_recommendations = list(set(recommended_movies_user) | set(recommended_movies_content))
    combined_recommendations = sorted(
        [(movie, similarity) for movie, similarity in combined_recommendations if movie != movie_title],
        key=lambda x: x[1], reverse=True)

    # Prediction Step for Collaborative Filtering
    unrated_movies = user_movie_ratings.loc[user_id][user_movie_ratings.loc[user_id] == 0].index
    predicted_ratings = []

    for movie, similarity in combined_recommendations:
        if movie in unrated_movies:
            # Calculate predicted rating based on similar user ratings for the movie
            similar_movie_ratings = user_movie_ratings[movie]
            predicted_rating = (similar_movie_ratings * distances_user).sum() / distances_user.sum()
            predicted_ratings.append((movie, similarity, predicted_rating))

    # Sort by predicted rating in descending order
    predicted_ratings = sorted(predicted_ratings, key=lambda x: x[2], reverse=True)

    # Display the filtered recommendations with predicted ratings
    st.subheader("Your personalized Recommended Movies with Predicted Ratings (Collaborative Filtering)")
    for movie, similarity, predicted_rating in predicted_ratings[:12]:
        st.write(f"{movie}, Similarity: {similarity:.4f}, Predicted Rating: {predicted_rating:.4f}")

    st.divider()

    # Display the filtered recommendations with similarity scores for Content-Based Filtering
    st.subheader("Your personalized Recommended Movies (Content-Based Filtering)")
    for movie, similarity in recommended_movies_content[:12]:
        st.write(f"{movie}, Similarity: {similarity:.4f}")

    st.divider()

    # Allow users to provide feedback
    user_feedback = st.selectbox("How would you rate the recommendations?", [5, 4, 3, 2, 1])

    # You can save the feedback in your dataset for future model improvement
    st.write(f"Thank you for your feedback! You rated the recommendations as: {user_feedback}")
