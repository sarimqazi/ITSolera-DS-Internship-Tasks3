import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Netflix Movie Recommender", page_icon="🎬", layout="wide")

# Custom CSS for premium aesthetics
st.markdown("""
<style>
/* Dark Mode Colors & Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    color: #E50914; /* Netflix Red */
    font-weight: 700;
}

/* Glassmorphism Cards for Movie Recommendations */
.movie-card {
    background: rgba(30, 30, 30, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(229, 9, 20, 0.2);
    border: 1px solid rgba(229, 9, 20, 0.5);
}

.movie-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 5px;
    color: #ffffff;
}

.similarity-score {
    display: inline-block;
    background: linear-gradient(135deg, #E50914, #B20710);
    color: white;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.movie-meta {
    font-size: 0.9rem;
    color: #aaaaaa;
    margin-bottom: 10px;
}

.movie-desc {
    font-size: 1rem;
    color: #dddddd;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# Application Title & Description
st.title("🎬 Netflix Movie & TV Show Recommender")
st.markdown("Discover top recommendations based on your favorite movies and shows! Our system uses Content-Based Filtering (Cosine Similarity) on Netflix data to find items related to what you love.")

@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    return df

@st.cache_data
def preprocess_data(df):
    # Fill missing values
    df['director'] = df['director'].fillna('')
    df['cast'] = df['cast'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # Clean text to improve feature extraction
    def clean_text(text):
        return str(text).replace(",", " ").lower()

    df['combined_features'] = df['director'].apply(clean_text) + " " + \
                              df['cast'].apply(clean_text) + " " + \
                              df['listed_in'].apply(clean_text) + " " + \
                              df['description'].apply(clean_text)
    
    return df

@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Compute the cosine similarity matrix directly
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

@st.cache_data
def get_indices(df):
    return pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, df, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # If there are duplicates, grab the first one
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies (ignoring the first one which is the movie itself)
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    # Return the top 5 most similar movies and scores
    recommended_df = df.iloc[movie_indices].copy()
    recommended_df['similarity'] = scores
    return recommended_df

try:
    with st.spinner('Loading Netflix Dataset...'):
        df = load_data()
    
    with st.spinner('Preprocessing features...'):
        df = preprocess_data(df)
        
    with st.spinner('Building Recommendation Engine (TF-IDF & Cosine Similarity)... \\nNote: This may take a minute on the first run.'):
        cosine_sim = build_model(df)
        indices = get_indices(df)

    st.success("Recommendation engine ready!")
    
    # Sort movie list alphabetically for better UX
    movie_list = sorted(df['title'].tolist())
    
    st.markdown("### Select a Movie or TV Show")
    selected_movie = st.selectbox(
        "Search for a title...", 
        movie_list, 
        index=None,
        placeholder="Choose a movie..."
    )
    
    if st.button("Get Recommendations"):
        if selected_movie:
            with st.spinner(f"Finding similar titles to '{selected_movie}'..."):
                recommendations = get_recommendations(selected_movie, cosine_sim, df, indices)
                
            st.markdown(f"## Top 5 Recommendations for **{selected_movie}**")
            
            for index, row in recommendations.iterrows():
                similarity_percentage = round(row['similarity'] * 100, 1)
                
                # HTML template for the card
                card_html = f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="similarity-score">🔥 {similarity_percentage}% Match</div>
                    <div class="movie-meta">
                        <strong>Type:</strong> {row['type']} &nbsp;|&nbsp; 
                        <strong>Year:</strong> {row['release_year']} &nbsp;|&nbsp; 
                        <strong>Genres:</strong> {row['listed_in']}
                    </div>
                    <div class="movie-desc">{row['description']}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.warning("Please choose a movie or TV show first to get recommendations!")

except Exception as e:
    st.error(f"An error occurred: {e}")
