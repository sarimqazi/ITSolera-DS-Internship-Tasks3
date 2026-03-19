# Task 2: Netflix Movie Recommendation System

## Objective
Build a content-based recommendation system with a GUI for movies and TV shows using the Netflix dataset.

## Features
- **Similarity Algorithm:** Utilizes Natural Language Processing (NLP) with `TfidfVectorizer` and Cosine Similarity to recommend titles based on director, cast, description, and listed genres.
- **Interactive GUI:** Built with Streamlit, offering a premium and responsive user experience.
- **Smart Selection:** Users can search for and select movies from a pre-populated list sorted alphabetically.
- **Visual Recommendations:** The top 5 recommendations are displayed with similarity scores (match %), release year, genres, and an overview.

## Skills Gained
- Recommendation systems
- Similarity algorithms (Cosine Similarity, TF-IDF)
- Advanced GUI integration with Streamlit
- Custom CSS styling and responsive design

## How to Run
1. Ensure you have Python installed.
2. Navigate to the `Task2` directory in your terminal:
   ```bash
   cd Task2
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open the displayed Local URL in your web browser (typically `http://localhost:8501`).

## Usage
- Click the dropdown to search for and select a movie or TV show.
- Click the **Get Recommendations** button to generate top matches.
- A warning prompts you if the button is clicked without any title selected.
