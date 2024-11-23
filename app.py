import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Factor:
    def __init__(self, vars, table):
        self.vars = vars
        self.table = table
        if len(vars) != len(table.shape):
            raise ValueError(f"Variables {vars} do not match table dimensions {table.shape}")

    def multiply(self, other):
        common_vars = list(set(self.vars) & set(other.vars))
        if not common_vars:
            raise ValueError("Factors have no variables in common to multiply.")

        all_vars = self.vars + [var for var in other.vars if var not in self.vars]
        self_indices = {var: idx for idx, var in enumerate(self.vars)}
        other_indices = {var: idx for idx, var in enumerate(other.vars)}
        all_indices = {var: idx for idx, var in enumerate(all_vars)}

        result_shape = [
            self.table.shape[self_indices[var]] if var in self.vars else other.table.shape[other_indices[var]]
            for var in all_vars
        ]

        self_table_expanded = self.table
        other_table_expanded = other.table

        for var in all_vars:
            if var not in self.vars:
                axis = all_indices[var]
                self_table_expanded = np.expand_dims(self_table_expanded, axis=axis)
            if var not in other.vars:
                axis = all_indices[var]
                other_table_expanded = np.expand_dims(other_table_expanded, axis=axis)

        result_table = self_table_expanded * other_table_expanded
        return Factor(all_vars, result_table)

    def condition(self, evidence):
        conditioned_table = self.table
        for var, value in evidence.items():
            if var in self.vars:
                idx = self.vars.index(var)
                if value >= conditioned_table.shape[idx] or value < 0:
                    raise IndexError(f"Value {value} for variable {var} is out of bounds.")
                conditioned_table = np.take(conditioned_table, indices=value, axis=idx)
            else:
                raise KeyError(f"Variable {var} not found in factor variables: {self.vars}")
        remaining_vars = [var for var in self.vars if var not in evidence]
        return Factor(remaining_vars, conditioned_table)

    def normalize(self):
        total = self.table.sum()
        if total != 0:
            self.table = self.table / total


# Load
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


ratings_file_path = "./ratings_movies.csv"  

ratings_data = load_data(ratings_file_path)

rating_scale = np.arange(0, 5.5, 0.5)

train_data, test_data = train_test_split(ratings_data, test_size=0.2, random_state=42)

user_movie_counts = train_data.groupby(['userId', 'movieId']).size().reset_index(name='count')
total_ratings_per_user = user_movie_counts.groupby('userId')['count'].sum().reset_index()
total_ratings_per_user.columns = ['userId', 'total_count']

user_movie_counts = pd.merge(user_movie_counts, total_ratings_per_user, on='userId')
user_movie_counts['probability'] = user_movie_counts['count'] / user_movie_counts['total_count']

user_ids = user_movie_counts['userId'].unique()
movie_ids = user_movie_counts['movieId'].unique()
user_mapping = {user: idx for idx, user in enumerate(user_ids)}
movie_mapping = {movie: idx for idx, movie in enumerate(movie_ids)}

row_indices = user_movie_counts['movieId'].map(movie_mapping).fillna(-1).astype(int)
col_indices = user_movie_counts['userId'].map(user_mapping).fillna(-1).astype(int)

valid_rows = (row_indices >= 0) & (col_indices >= 0)
row_indices = row_indices[valid_rows]
col_indices = col_indices[valid_rows]
data = user_movie_counts.loc[valid_rows, 'probability']

prob_movie_given_user = Factor(
    ['Movie', 'User'],
    np.zeros((len(movie_ids), len(user_ids)))  
)

for i, j, val in zip(row_indices, col_indices, data):
    prob_movie_given_user.table[i, j] = val

movie_rating_counts = train_data.groupby(['movieId', 'rating']).size().reset_index(name='count')
total_ratings_per_movie = movie_rating_counts.groupby('movieId')['count'].sum().reset_index()
movie_rating_counts = pd.merge(movie_rating_counts, total_ratings_per_movie, on='movieId')
movie_rating_counts['probability'] = movie_rating_counts['count_x'] / movie_rating_counts['count_y']

row_indices = movie_rating_counts['rating'].astype('category').cat.codes
col_indices = movie_rating_counts['movieId'].map(movie_mapping).fillna(-1).astype(int)

valid_rows = col_indices >= 0
row_indices = row_indices[valid_rows]
col_indices = col_indices[valid_rows]
data = movie_rating_counts.loc[valid_rows, 'probability']

prob_rating_given_movie = Factor(
    ['Rating', 'Movie'],
    np.zeros((len(rating_scale), len(movie_ids)))  
)

for i, j, val in zip(row_indices, col_indices, data):
    prob_rating_given_movie.table[i, j] = val

joint_factor = prob_rating_given_movie.multiply(prob_movie_given_user)

st.title("Movie Rating Prediction")

st.subheader("Test Dataset Overview")
st.dataframe(test_data[['userId', 'movieId', 'rating']])

user_id = st.selectbox("Select User ID:", sorted(test_data['userId'].unique()))
movie_id = st.selectbox("Select Movie ID:", sorted(test_data['movieId'].unique()))

if st.button("Predict Rating"):
    if user_id in user_mapping and movie_id in movie_mapping:
        evidence = {
            "User": user_mapping.get(user_id, -1),
            "Movie": movie_mapping.get(movie_id, -1),
        }
        conditioned_factor = joint_factor.condition(evidence)
        conditioned_factor.normalize()

        predicted_index = np.argmax(conditioned_factor.table)
        predicted_rating = rating_scale[predicted_index]
        st.success(f"Predicted Rating: {predicted_rating}")

        # Visualize 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(rating_scale, conditioned_factor.table, width=0.4, align="center")
        ax.set_xlabel("Ratings")
        ax.set_ylabel("Probability")
        ax.set_title("Probability Distribution of Ratings")
        st.pyplot(fig)
    else:
        st.error("Invalid User ID or Movie ID.")
