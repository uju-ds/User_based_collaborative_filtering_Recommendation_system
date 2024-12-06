import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
import typing
from typing import *

rating_df = pd.read_csv(r'C:\Users\user\Desktop\ratings.csv')
movies_df = pd.read_csv(r'C:\Users\user\Desktop\movies.csv')

# merge the two datasets 
merge_df = pd.merge(rating_df,movies_df,how="inner",on="movieId")
print(merge_df.head())

# tranform the data to user-movies matrix
def transform_csv_dataframe_to_user_movies_matrix(csv_df: DataFrame) -> DataFrame:
  user_movie_matrix = csv_df.pivot(index='userId', columns='movieId', values='rating')
  user_movie_matrix.reset_index(inplace=True)
  user_movie_matrix.columns.name = None
  return user_movie_matrix

#User-based Collaborative Filtering using Pearson Similarity

def pearson_similarity(user1_ratings: Series, user2_ratings: Series, **kwargs) -> float:
    common_ratings = user1_ratings.dropna().index.intersection(user2_ratings.dropna().index) #Finds movies rated by both users.
    if len(common_ratings) == 0:
        return np.nan
    user1_common = user1_ratings[common_ratings]
    user2_common = user2_ratings[common_ratings]
#Subtract each user's mean rating to center their ratings.
    mean_user1 = user1_common.mean()
    mean_user2 = user2_common.mean()
    numerator = ((user1_common - mean_user1) * (user2_common - mean_user2)).sum()
    denominator = np.sqrt(((user1_common - mean_user1) ** 2).sum()) * np.sqrt(((user2_common - mean_user2) ** 2).sum())
    with np.errstate(invalid='ignore'):
        return numerator / denominator
    
#Prepare User-Movie Matrix for Collaborative Filtering
def prepare_user_movie_matrix_for_cf_matrix(
    user_movie_matrix: DataFrame,
    movie_id: int,
    user_id: int,
  ) -> DataFrame:
#Fetch the target user's row and drop movies they haven't rated.
    user_ratings = user_movie_matrix[user_movie_matrix['userId'] == user_id].dropna(axis=1, how='all')
    
    rated_movies = user_ratings.columns[1:]
    if movie_id in rated_movies:
        raise ValueError(f"The user {user_id} has already rated the movie {movie_id}")
    
#Identify users who have rated at least one of the same movies as the target user and the target movie (movie_id).    
    relevant_users = user_movie_matrix[user_movie_matrix[rated_movies].notna().any(axis=1) & user_movie_matrix[movie_id].notna()]
    
    rated_movies = rated_movies.append(pd.Index([movie_id]))

# Concatenate the target user's row with rows of relevant users and the resulting DataFrame is the prepared matrix for collaborative filtering.

    relevant_users_rows = relevant_users[['userId'] + list(rated_movies)]
    user_row = user_movie_matrix[user_movie_matrix["userId"]==user_id][['userId'] + list(rated_movies)]
    return pd.concat([user_row, relevant_users_rows]).reset_index(drop=True)

#Add Similarity Column to CF Matrix

def add_similarity_column_to_cf_matrix(
    cf_matrix: DataFrame, movie_id: int, user_id: int,
    similarity_measure: typing.Callable[[Series, Series, Any], np.float64],
  ) -> DataFrame:
#The ratings of the target user, excluding the movie to be predicted, are extracted.    
    target_user_ratings = cf_matrix[cf_matrix['userId'] == user_id].drop('userId', axis=1).drop(movie_id, axis=1).iloc[0]
    average_non_nan = cf_matrix[cf_matrix['userId'] != user_id].drop(['userId', movie_id], axis=1).notna().sum(axis=1).mean()
    similarities = []
    for other_user_id in cf_matrix['userId']:
        if other_user_id == user_id:
            similarities.append(1.0)
        else:
            other_user_ratings = cf_matrix[cf_matrix['userId'] == other_user_id].drop('userId', axis=1).drop(movie_id, axis=1).iloc[0]
            similarity = similarity_measure(target_user_ratings, other_user_ratings, average_non_nan=average_non_nan)
            similarities.append(similarity)
    df_copy = cf_matrix.copy()
    df_copy['similarity'] = similarities
    return df_copy
#This function filters the collaborative filtering matrix to include only relevant users (neighbors).
def limit_neighborhood(
    cf_matrix_with_similarity: DataFrame,
    movie_id: int,
    user_id: int,
    threshold: Optional[float] = None,
) -> DataFrame:
    neighbors = cf_matrix_with_similarity.copy()
    
    if threshold is not None:
        neighbors = neighbors[neighbors['similarity'] >= threshold]

    if len(neighbors) < 3:
        raise ValueError("Number of neighbors is less than 1")

    return neighbors

#This function predicts a rating for a target user and movie based on the ratings of the neighbors.
def predict_score(cf_matrix_with_similarity: DataFrame, movie_id: int, user_id: int) -> float:
    neighbors_ratings = cf_matrix_with_similarity[
        cf_matrix_with_similarity['userId'] != user_id
    ][['userId', movie_id, 'similarity']].dropna()

    target_user_ratings = cf_matrix_with_similarity[
        cf_matrix_with_similarity['userId'] == user_id
    ].drop(['userId', 'similarity'], axis=1).iloc[0].dropna()
    target_user_avg_rating = target_user_ratings.mean()

    neighbors_avg_ratings = cf_matrix_with_similarity[
        cf_matrix_with_similarity['userId'].isin(neighbors_ratings['userId'])
    ].drop(['userId', 'similarity'], axis=1).mean(axis=1)

#Computes a weighted sum using neighbors' deviations from their average rating and their similarity.
    weighted_sum = sum(
        (neighbors_ratings.iloc[i][movie_id] - neighbors_avg_ratings.iloc[i]) * neighbors_ratings.iloc[i]['similarity']
        for i in range(len(neighbors_ratings))
    )

    sum_of_similarities = neighbors_ratings['similarity'].sum()

    predicted_score = target_user_avg_rating + (weighted_sum / sum_of_similarities)

    return predicted_score
     

#Calculate Prediction using Pearson Similarity

def calculate_prediction_using_pearson_similarity(
    user_id: int, 
    movie_id: int
) -> float:
    THRESHOLD = None
    transformed_df = transform_csv_dataframe_to_user_movies_matrix(merge_df) #Converts the original dataset into a user-movie matrix.
#Selects relevant rows and columns for collaborative filtering.
    limited_df = prepare_user_movie_matrix_for_cf_matrix(
        transformed_df, movie_id, user_id
    )
    limited_df_with_similarity = add_similarity_column_to_cf_matrix(
        limited_df, movie_id, user_id, pearson_similarity
    )
    neighbors_df = limit_neighborhood(
        limited_df_with_similarity,
        movie_id,
        user_id,
        THRESHOLD,
    )
    predicted_score = predict_score(
        neighbors_df,
        movie_id,
        user_id,
    )
    return predicted_score



# Define the main function
def main():

    print("Welcome to the Movie Recommendation System!")
    try:
        user_id = int(input("Enter the user ID: "))
        movie_id = int(input("Enter the movie ID: "))
    except ValueError:
        print("Invalid input. Please enter integer values for user ID and movie ID.")
        return

    if movie_id not in merge_df['movieId'].unique():
        print(f"Error: Movie ID {movie_id} not found in the dataset.")
        return
      # Fetch the movie name using the movie_id
    movie_name = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    
    user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(merge_df)

    try:
        cf_matrix = prepare_user_movie_matrix_for_cf_matrix(user_movie_matrix, movie_id, user_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    try:
        cf_matrix_with_similarity = add_similarity_column_to_cf_matrix(
            cf_matrix, movie_id, user_id, pearson_similarity
        )
    except Exception as e:
        print(f"Error while calculating similarities: {e}")
        return

    try:
        neighbors_df = limit_neighborhood(cf_matrix_with_similarity, movie_id, user_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    try:
        predicted_score = predict_score(neighbors_df, movie_id, user_id)
        print(f"\nPredicted rating for User {user_id} on Movie '{movie_name}' (Movie ID: {movie_id}): {predicted_score:.2f}")
    except Exception as e:
        print(f"Error while predicting the score: {e}")
        return

if __name__ == "__main__":
    main()
