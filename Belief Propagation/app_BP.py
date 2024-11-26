import streamlit as st
import pickle
with open('predicted_beliefs.pkl', 'rb') as f:
    predicted_beliefs = pickle.load(f)

with open('user_movie_matrix.pkl', 'rb') as f:
    user_movie_matrix = pickle.load(f)
st.title("Movie Rating Prediction")
st.write("This app predicts a user's rating for a movie using belief propagation.")

user_id = st.selectbox("Select User ID", options=user_movie_matrix.index.tolist())
movie_id = st.selectbox("Select Movie ID", options=user_movie_matrix.columns.tolist())

if user_id and movie_id:
    predicted_rating = predicted_beliefs[user_id].get(movie_id, 0)
    st.write(f"Predicted rating for user {user_id} on movie {movie_id}: {predicted_rating}")

