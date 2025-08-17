# --- Import Libraries ---
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page Setup ---
st.set_page_config(layout="wide")
st.header("📚 Book Recommender System")

st.markdown("""
##### 🔍 This site uses collaborative filtering to suggest books.  
##### 📖 You can also explore the **Top 50 Books** in our dataset.  
""")

# --- Cache Helpers ---
@st.cache_data
def load_pickle(filename):
    return pickle.load(open(filename, "rb"))

@st.cache_data
def load_csv(filename):
    return pd.read_csv(filename, low_memory=False)

# --- Load Pickled Data ---
popular = load_pickle("popular.pkl")
books = load_pickle("books.pkl")
pt = load_pickle("pt.pkl")
similarity_scores = load_pickle("similarity_scores.pkl")

# --- Helpers to detect correct columns ---
def get_col(df, keyword):
    for col in df.columns:
        if keyword in col.lower():
            return col
    return None

# Detect column names dynamically
popular_img_col   = get_col(popular, "image")
books_img_col     = get_col(books, "image")

# Fallbacks for Book-Crossing dataset defaults
if not popular_img_col:
    for fallback in ["Image-URL-M", "Image-URL-L", "Image-URL-S"]:
        if fallback in popular.columns:
            popular_img_col = fallback
            break

if not books_img_col:
    for fallback in ["Image-URL-M", "Image-URL-L", "Image-URL-S"]:
        if fallback in books.columns:
            books_img_col = fallback
            break

# Title & Author columns
popular_title_col = get_col(popular, "title")
popular_author_col= get_col(popular, "author")
books_title_col   = get_col(books, "title")
books_author_col  = get_col(books, "author")

# --- Top 50 Books Section ---
st.sidebar.title("📊 Explore")

if st.sidebar.button("Show Top 50 Books"):
    cols_per_row = 5
    num_rows = 10
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            book_idx = row * cols_per_row + col
            if book_idx < len(popular):
                with cols[col]:
                    if popular_img_col:
                        st.image(popular.iloc[book_idx][popular_img_col], use_container_width=True)
                    if popular_title_col:
                        st.text(popular.iloc[book_idx][popular_title_col])
                    if popular_author_col:
                        st.text(popular.iloc[book_idx][popular_author_col])

# --- Recommend Books Function ---
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    data = []
    for i in similar_items:
        temp_df = books[books[books_title_col] == pt.index[i[0]]].drop_duplicates(books_title_col)
        if not temp_df.empty:
            item = [
                temp_df[books_title_col].values[0],
                temp_df[books_author_col].values[0],
                temp_df[books_img_col].values[0] if books_img_col else ""
            ]
            data.append(item)
    return data

# --- Similar Book Suggestions ---
book_list = pt.index.values
selected_book = st.sidebar.selectbox("🔎 Select a book to get recommendations", book_list)

if st.sidebar.button("Recommend Me"):
    book_recommend = recommend(selected_book)
    cols = st.columns(5)
    for col_idx, book in enumerate(book_recommend):
        with cols[col_idx]:
            if book[2] != "":
                st.image(book[2], use_container_width=True)
            st.text(book[0])  # title
            st.text(book[1])  # author

# --- Show Dataset (On Demand) ---
if st.sidebar.button("Show Raw Data"):
    st.subheader("📕 Books Data")
    st.dataframe(load_csv("Books.csv"))

    st.subheader("⭐ Ratings Data")
    st.dataframe(load_csv("Ratings.csv"))

    st.subheader("👤 Users Data")
    st.dataframe(load_csv("Users.csv"))
