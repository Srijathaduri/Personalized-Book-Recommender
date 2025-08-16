# import the libraries 
import streamlit as st 
import pickle 
import numpy as np 
import pandas as pd 

st.set_page_config(layout="wide")

st.header("📚 Book Recommender System")

st.markdown('''
##### The site using collaborative filtering suggests books from our catalog. 
##### We recommend top 50 books for everyone as well. 
''')

# --- Load pickled models/datasets ---
popular = pickle.load(open('popular.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb')) 

# --- Helpers to detect correct columns ---
def get_col(df, keyword):
    for col in df.columns:
        if keyword in col.lower():
            return col
    return None

# Detect column names dynamically (with fallback)
popular_img_col   = get_col(popular, "image")
books_img_col     = get_col(books, "image")

# Force fallback to Book-Crossing dataset defaults
if not popular_img_col:
    if "Image-URL-M" in popular.columns:
        popular_img_col = "Image-URL-M"
    elif "Image-URL-L" in popular.columns:
        popular_img_col = "Image-URL-L"
    elif "Image-URL-S" in popular.columns:
        popular_img_col = "Image-URL-S"

if not books_img_col:
    if "Image-URL-M" in books.columns:
        books_img_col = "Image-URL-M"
    elif "Image-URL-L" in books.columns:
        books_img_col = "Image-URL-L"
    elif "Image-URL-S" in books.columns:
        books_img_col = "Image-URL-S"

# Title & Author columns
popular_title_col = get_col(popular, "title")
popular_author_col= get_col(popular, "author")
books_title_col   = get_col(books, "title")
books_author_col  = get_col(books, "author")

# --- Top 50 Books section ---
st.sidebar.title("Top 50 Books")

if st.sidebar.button("SHOW"):
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

# --- Function to recommend books ---
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books[books_title_col] == pt.index[i[0]]].drop_duplicates(books_title_col)
        
        item.extend(temp_df[books_title_col].values)
        item.extend(temp_df[books_author_col].values)
        
        if books_img_col:
            item.extend(temp_df[books_img_col].values)
        else:
            item.append("")  # empty if no image
        data.append(item) 
    return data

# --- Book suggestion section ---
book_list = pt.index.values 

st.sidebar.title("Similar Book Suggestions")
selected_book = st.sidebar.selectbox("Select a book from the dropdown", book_list)

if st.sidebar.button("Recommend Me"):
    book_recommend = recommend(selected_book)
    cols = st.columns(5)
    for col_idx in range(5):
        with cols[col_idx]:
            if col_idx < len(book_recommend):
                if book_recommend[col_idx][2] != "":
                    st.image(book_recommend[col_idx][2], use_container_width=True)
                st.text(book_recommend[col_idx][0])
                st.text(book_recommend[col_idx][1])

# --- Raw data for inspection ---
books_df   = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\PersonalizeBookRecomender\Books.csv")
users_df   = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\PersonalizeBookRecomender\Users.csv")
ratings_df = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\PersonalizeBookRecomender\Ratings.csv")

st.sidebar.title("Data Used")

if st.sidebar.button("Show"):
    st.subheader('This is the books data we used in our model')
    st.dataframe(books_df)
    st.subheader('This is the User ratings data we used in our model')
    st.dataframe(ratings_df)
    st.subheader('This is the user data we used in our model')
    st.dataframe(users_df)
