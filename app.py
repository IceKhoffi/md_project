import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Netflix TV Show Recommendation", layout="wide")

# Your other Streamlit code can follow
st.title("Netflix TV Show Recommendation System")

# Load pickled data
@st.cache_data
def load_data():
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vector = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("cosine_similarity.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    return df, tfidf_vector, tfidf_matrix, cosine_sim


df, tfidf_vector, tfidf_matrix, cosine_sim = load_data()
titles = df['title'].unique().tolist()

# Mapping title -> index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def recommend(title, top_n=5):
    try:
        idx = indices[title]
    except KeyError:
        return "Title not found in dataset."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [score for score in sim_scores if score[0] != idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    rec_indices = [i[0] for i in sim_scores[:top_n]]

    result_df = df.iloc[rec_indices][['title', 'listed_in', 'description']]
    return result_df

# Session state for navigation
if "selected_title" not in st.session_state:
    st.session_state.selected_title = None
if "view" not in st.session_state:
    st.session_state.view = "search"

# BACK BUTTON
if st.session_state.view != "search":
    if st.button("🔙 Back"):
        st.session_state.view = "search"
        st.session_state.selected_title = None

# SEARCH VIEW
if st.session_state.view == "search":
    user_input = st.selectbox("Enter a TV Show Title:", options=titles)
    if st.button("Get Recommendations"):
        st.session_state.selected_title = user_input
        st.session_state.view = "details"

# DETAILS VIEW
if st.session_state.view == "details" and st.session_state.selected_title:
    title = st.session_state.selected_title
    show_info = df[df['title'] == title].iloc[0]
    
    st.subheader(f"🎬 {title}")
    st.markdown(f"**Genre**: {show_info['listed_in']}")
    st.markdown(f"**Description**: {show_info['description']}")

    # Show Recommendations
    st.markdown("### 📌 Other Recommended TV Shows:")
    recs = recommend(title)
    if isinstance(recs, str):
        st.error(recs)
    else:
        for _, row in recs.iterrows():
            if st.button(row['title']):
                st.session_state.selected_title = row['title']
