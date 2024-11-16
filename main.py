import pandas as pd
import numpy as np

# Load the MovieLens dataset
ratings_df = pd.read_csv(r"C:\Users\HP\Downloads\ml-latest-small\ml-latest-small\ratings.csv")
movies_df = pd.read_csv(r"C:\Users\HP\Downloads\ml-latest-small\ml-latest-small\movies.csv")


# Explore the data
print(ratings_df.head())
print(movies_df.head())

# Merge ratings and movies dataframes
merged_df = ratings_df.merge(movies_df, on='movieId')

# Create a user-item matrix
user_item_matrix = merged_df.pivot_table(index='userId', columns='title', values='rating')

# Handle missing values (e.g., fill with 0)
user_item_matrix.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity

# Calculate user-based similarity
user_similarity = cosine_similarity(user_item_matrix)

# Function to recommend movies based on user similarity
def recommend_movies(user_id, user_similarity, user_item_matrix, num_recommendations=10):
    similar_users = user_similarity[user_id]
    similar_users = similar_users.argsort()[:-num_recommendations-1:-1]

    recommendations = user_item_matrix.iloc[similar_users].mean(axis=0) - user_item_matrix.iloc[user_id]
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations.head(num_recommendations)

# Example usage
recommended_movies = recommend_movies(1, user_similarity, user_item_matrix)
print(recommended_movies)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a TF-IDF matrix based on movie titles and genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['title'] + ' ' + movies_df['genres'])

# Calculate cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on content similarity
def recommend_movies_content_based(title, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

def recommend_movies_content_based(title, cosine_sim=cosine_sim):
  if title not in movies_df['title'].tolist():
    print(f"Movie '{title}' not found in the dataset.")
    return None
  idx = movies_df[movies_df['title'] == title].index[0]
  # ... rest of the function logic

# Example usage
recommended_movies = recommend_movies_content_based('The Shawshank Redemption')
print(recommended_movies)