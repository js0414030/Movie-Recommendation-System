Movie Recommendation System

  This project implements a movie recommendation system using the MovieLens dataset. The system provides personalized movie recommendations through two techniques:
  

Collaborative Filtering (User-Based): This method leverages user similarity to recommend movies based on ratings from similar users.

Content-Based Filtering: This method uses a TF-IDF vectorizer to analyze movie titles and genres, recommending similar movies based on content similarity.


Features:

  - Loads and merges movie ratings and metadata.
  
  - Creates a user-item matrix for collaborative filtering.
  
  - Calculates user similarity using cosine similarity.
  
  - Implements a function to recommend movies based on user preferences.
  
  - Uses TF-IDF vectorization to recommend movies based on title and genre similarity.
  

Limitations:

  - Currently only supports user-based and content-based filtering.
  
  - Dataset size may limit the quality of recommendations for smaller datasets.
  
  - The recommendation model can be improved with more sophisticated techniques like matrix factorization.
  
  
Future Updates:

  - Implement item-based collaborative filtering.
  
  - Add deep learning models for better accuracy.
  
  - Support real-time recommendations based on dynamic user behavior.
