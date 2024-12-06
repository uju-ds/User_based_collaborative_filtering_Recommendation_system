# User_based_collaborative_filtering_Recommendation_system
Collaborative Filtering and Group Recommendations in MovieLens 100K
The steps involved are;
- transform_csv_dataframe_to_user_movies_matrix
 - pearson_similarity
 - prepare_user_movie_matrix_for_cf_matrix
 - add_similarity_column_to_cf_matrix
 - limit_neighborhood
 - predict_score
 - calculate_prediction_using_pearson_similarity

- Transform CSV DataFrame to User-Movies Matrix
The transform_csv_dataframe_to_user_movies_matrix function transforms the DataFrame resulted from the dataset CSV into a user-movies matrix where the rows represent users, the columns represent movies, and the values represent the ratings given by users to movies.

- Prepare User-Movie Matrix for Collaborative Filtering
The prepare_user_movie_matrix_for_cf_matrix function prepares the user-movie matrix for collaborative filtering by filtering the matrix to include only the relevant users and movies. It retrieves the movies rated by the specific user and filters the user-movie matrix to include only the relevant users and movies.

- Add Similarity Column to CF Matrix
The add_similarity_column_to_cf_matrix function adds a new column to the collaborative filtering (CF) user-movie matrix that represents the similarity between the target user and each user in the matrix. It calculates the similarity using the provided similarity_measure function.

- Predict Score
The predict_score function calculates the predicted score for a movie by a user based on the collaborative filtering matrix with similarities. It retrieves the neighbors' ratings for the specified movie, calculates the average rating of the target user, and the average rating of each neighbor. It then calculates the weighted sum of the neighbors' ratings and the sum of the similarities to predict the score.

- Calculate Prediction using Pearson Similarity
The calculate_prediction_using_pearson_similarity function calculates the predicted score for a user-movie pair using the Pearson similarity. It transforms the raw DataFrame into a user-movie matrix, prepares the matrix for collaborative filtering, adds the similarity column to the matrix, limits the neighborhood based on a threshold, and predicts the score using the collaborative filtering matrix.
