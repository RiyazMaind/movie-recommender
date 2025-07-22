import pandas as pd
import pickle
from scipy import sparse
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS for cross-origin requests
import random # Import random for selecting a featured movie
import os # Import os for path manipulation

# Initialize Flask app
app = Flask(__name__) # Ensure the app instance is named 'app'
CORS(app) # Enable CORS for all routes, allowing your frontend to access it

# Global variables to store the loaded models and data
df = None
tfidf = None
tfidf_matrix = None
cosine_sim = None
indices = None
models_initialized = False # New flag to track if models loaded successfully

def load_models():
    """
    Loads the pre-trained models and data.
    This function should be called once when the application starts.
    """
    global df, tfidf, tfidf_matrix, cosine_sim, indices, models_initialized
    try:
        # Define base path for models. Assumes models are in the same directory as app.py
        # If your models are in a subfolder (e.g., 'data/'), change this:
        # base_path = os.path.join(os.path.dirname(__file__), 'data')
        base_path = os.path.dirname(__file__) # Path to the directory where app.py resides

        # Load the DataFrame
        df_path = os.path.join(base_path, 'movies_df.pkl')
        print(f"Attempting to load DataFrame from: {df_path}")
        df = pd.read_pickle(df_path)
        df['title_lower'] = df['title'].str.lower()
        indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()
        print("DataFrame loaded successfully.")

        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
        print(f"Attempting to load TF-IDF vectorizer from: {tfidf_path}")
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        print("TF-IDF vectorizer loaded successfully.")

        # Load TF-IDF matrix (sparse)
        tfidf_matrix_path = os.path.join(base_path, 'tfidf_matrix.npz')
        print(f"Attempting to load TF-IDF matrix from: {tfidf_matrix_path}")
        tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
        print("TF-IDF matrix loaded successfully.")

        # Load cosine similarity matrix
        cosine_sim_path = os.path.join(base_path, 'cosine_sim.npy')
        print(f"Attempting to load cosine similarity matrix from: {cosine_sim_path}")
        cosine_sim = np.load(cosine_sim_path)
        print("Cosine similarity matrix loaded successfully.")

        models_initialized = True
        print("All models and data loaded successfully and initialized flag set to True!")

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: One or more model files not found. Please ensure they are in the correct directory. Error: {e}")
        models_initialized = False
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred during model loading: {e}")
        models_initialized = False

def _process_movie_data(movie_data):
    """
    Helper function to process movie data, converting string fields to lists
    and adding mock trailer URLs.
    """
    # Ensure genres, production_companies, etc., are lists if they are strings
    if isinstance(movie_data.get('genres'), str):
        movie_data['genres'] = [g.strip() for g in movie_data['genres'].split(',')]
    else: # Ensure it's always a list, even if empty or None
        movie_data['genres'] = movie_data.get('genres', []) if movie_data.get('genres') is not None else []

    if isinstance(movie_data.get('production_companies'), str):
        movie_data['production_companies'] = [pc.strip() for pc in movie_data['production_companies'].split(',')]
    else:
        movie_data['production_companies'] = movie_data.get('production_companies', []) if movie_data.get('production_companies') is not None else []

    if isinstance(movie_data.get('production_countries'), str):
        movie_data['production_countries'] = [pc.strip() for pc in movie_data['production_countries'].split(',')]
    else:
        movie_data['production_countries'] = movie_data.get('production_countries', []) if movie_data.get('production_countries') is not None else []

    if isinstance(movie_data.get('spoken_languages'), str):
        movie_data['spoken_languages'] = [sl.strip() for sl in movie_data['spoken_languages'].split(',')]
    else:
        movie_data['spoken_languages'] = movie_data.get('spoken_languages', []) if movie_data.get('spoken_languages') is not None else []

    # --- MOCK TRAILER URL FOR DEMONSTRATION ---
    # Replace this with actual logic to fetch trailer URLs from a real API or your dataset
    title_lower = movie_data.get('title', '').lower() # Use .get() for safety
    if "inception" in title_lower:
        movie_data['trailer_url'] = 'https://www.youtube.com/watch?v=YoHD9XEInc0'
    elif "the matrix" in title_lower:
        movie_data['trailer_url'] = 'https://www.youtube.com/watch?v=vKQi3bBA1y8'
    elif "interstellar" in title_lower:
        movie_data['trailer_url'] = 'https://www.youtube.com/watch?v=zSWdZVtXT7E'
    elif "the dark knight" in title_lower:
        movie_data['trailer_url'] = 'https://www.youtube.com/watch?v=EXeTwQWrcwY'
    else:
        movie_data['trailer_url'] = None
    # --- END MOCK ---
    return movie_data


def recommend(title, k=5):
    """
    Generates movie recommendations based on a given title.
    """
    if not models_initialized:
        print("Models not initialized. Cannot provide recommendations.")
        return []

    title_lower = title.lower()
    if title_lower not in indices:
        print(f"Movie '{title}' not found in the database for recommendations.")
        return []

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]  # Exclude the movie itself

    movie_indices = [i[0] for i in sim_scores]
    columns_to_select = [
        'id', 'title', 'vote_average', 'popularity', 'overview', 'poster_path', 'tagline',
        'genres', 'runtime', 'release_date', 'production_companies',
        'production_countries', 'spoken_languages'
    ]
    recommended_movies_df = df.iloc[movie_indices][columns_to_select]

    recommended_movies = recommended_movies_df.to_dict(orient='records')
    processed_movies = [_process_movie_data(movie) for movie in recommended_movies]
    return processed_movies

# Define the recommendation API endpoint
@app.route('/recommend', methods=['POST'])
def get_movie_recommendations():
    if not models_initialized:
        return jsonify({'error': 'Server models not ready. Please try again in a moment.'}), 503 # 503 Service Unavailable

    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({'error': 'Invalid request: "title" missing in JSON payload'}), 400

    movie_title = data['title']
    print(f"Received recommendation request for movie: {movie_title}")
    recommendations = recommend(movie_title)

    if not recommendations:
        return jsonify({'message': 'No recommendations found for this movie or movie not in database.'}), 200
    else:
        return jsonify({'recommendations': recommendations}), 200

# Endpoint for fetching movie details
@app.route('/movie_details/<title>', methods=['GET'])
def get_movie_details(title):
    if not models_initialized:
        return jsonify({'error': 'Server models not ready. Please try again in a moment.'}), 503

    title_lower = title.lower()
    movie_row = df[df['title_lower'] == title_lower]

    if not movie_row.empty:
        columns_to_select = [
            'id', 'title', 'overview', 'poster_path', 'vote_average', 'popularity', 'tagline',
            'genres', 'backdrop_path', 'runtime', 'release_date',
            'production_companies', 'production_countries', 'spoken_languages'
        ]
        details = movie_row.iloc[0][columns_to_select].to_dict()
        details = _process_movie_data(details)

        return jsonify({'details': details}), 200
    else:
        return jsonify({'message': 'Movie details not found.'}), 404

# Endpoint for autocomplete suggestions
@app.route('/movies', methods=['GET'])
def get_movie_titles():
    if not models_initialized:
        return jsonify({'error': 'Server models not ready. Please try again in a moment.'}), 503

    query = request.args.get('q', '').lower()
    if query:
        filtered_titles = df[df['title_lower'].str.contains(query, na=False)]['title'].tolist()
    else:
        filtered_titles = df['title'].drop_duplicates().head(50).tolist()

    return jsonify({'titles': filtered_titles}), 200

# Endpoint for a featured movie banner
@app.route('/featured_movie', methods=['GET'])
def get_featured_movie():
    if not models_initialized:
        return jsonify({'error': 'Server models not ready. Please try again in a moment.'}), 503

    eligible_movies = df[(df['backdrop_path'].notna()) & (df['popularity'] > 50)]

    if eligible_movies.empty:
        eligible_movies = df[df['backdrop_path'].notna()]
        if eligible_movies.empty:
            return jsonify({'message': 'No movies with backdrop paths available.'}), 404

    featured_movie = eligible_movies.sample(n=1).iloc[0]

    details = featured_movie[['title', 'overview', 'backdrop_path', 'tagline']].to_dict()
    details = _process_movie_data(details)
    return jsonify({'featured_movie': details}), 200

# Endpoint for fetching popular movies for the "Movies" page
@app.route('/movies/popular', methods=['GET'])
def get_popular_movies():
    if not models_initialized:
        return jsonify({'error': 'Server models not ready. Please try again in a moment.'}), 503

    popular_movies_df = df[df['poster_path'].notna()].sort_values(by='popularity', ascending=False).head(20)

    columns_to_select = [
        'id', 'title', 'vote_average', 'popularity', 'overview', 'poster_path', 'tagline',
        'genres', 'runtime', 'release_date', 'production_companies',
        'production_countries', 'spoken_languages'
    ]
    popular_movies = popular_movies_df[columns_to_select].to_dict(orient='records')

    processed_movies = [_process_movie_data(movie) for movie in popular_movies]
    return jsonify({'movies': processed_movies}), 200


# Run the Flask app
if __name__ == '__main__':
    # Call load_models() when the app starts.
    # This will attempt to load models and set models_initialized flag.
    load_models()
    # Gunicorn will typically handle running the Flask app.
    # The app.run() is mainly for local development.
    # For Render, the 'gunicorn app:app' command handles this.
    # app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000)) # Use Render's PORT env var