import pandas as pd
import pickle
from scipy import sparse
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS for cross-origin requests
import random # Import random for selecting a featured movie

# Initialize Flask app
app = Flask(__name__) # Ensure the app instance is named 'app'
CORS(app) # Enable CORS for all routes, allowing your frontend to access it

# Global variables to store the loaded models and data
df = None
tfidf = None
tfidf_matrix = None
cosine_sim = None
indices = None

def load_models():
    """
    Loads the pre-trained models and data.
    This function should be called once when the application starts.
    """
    global df, tfidf, tfidf_matrix, cosine_sim, indices
    try:
        # Load the DataFrame
        df = pd.read_pickle('movies_df.pkl')
        df['title_lower'] = df['title'].str.lower()
        # Drop duplicates based on title_lower, keeping the first occurrence
        # This handles cases where multiple movies have the same title
        indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()
        print("DataFrame loaded successfully.")

        # Load TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        print("TF-IDF vectorizer loaded successfully.")

        # Load TF-IDF matrix (sparse)
        tfidf_matrix = sparse.load_npz('tfidf_matrix.npz')
        print("TF-IDF matrix loaded successfully.")

        # Load cosine similarity matrix
        cosine_sim = np.load('cosine_sim.npy')
        print("Cosine similarity matrix loaded successfully.")

        print("All models and data loaded successfully!")

    except FileNotFoundError as e:
        print(f"Error: One or more model files not found. Please ensure they are in the same directory as app.py. {e}")
        # In a production environment, you might want to log this and gracefully fail
        # For deployment, it's critical these files are present.
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        exit(1)

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
    title_lower = movie_data['title'].lower()
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

    Args:
        title (str): The title of the movie to get recommendations for.
        k (int): The number of recommendations to return.

    Returns:
        list: A list of dictionaries, each representing a recommended movie.
    """
    if df is None or cosine_sim is None or indices is None:
        print("Models not loaded. Cannot provide recommendations.")
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
    # Select all relevant columns for the frontend display and modal
    columns_to_select = [
        'id', 'title', 'vote_average', 'popularity', 'overview', 'poster_path', 'tagline',
        'genres', 'runtime', 'release_date', 'production_companies',
        'production_countries', 'spoken_languages'
    ]
    recommended_movies_df = df.iloc[movie_indices][columns_to_select]

    recommended_movies = recommended_movies_df.to_dict(orient='records')
    # Process each movie to clean data and add mock trailer URLs
    processed_movies = [_process_movie_data(movie) for movie in recommended_movies]
    return processed_movies

# Define the recommendation API endpoint
@app.route('/recommend', methods=['POST'])
def get_movie_recommendations():
    """
    API endpoint to receive a movie title and return recommendations.
    Expects a JSON payload with a 'title' key.
    """
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
    """
    API endpoint to fetch details for a specific movie title.
    """
    if df is None:
        return jsonify({'error': 'Models not loaded. Cannot fetch movie details.'}), 500

    title_lower = title.lower()
    # Find the movie in the DataFrame
    movie_row = df[df['title_lower'] == title_lower]

    if not movie_row.empty:
        # Get the first matching row and convert to dictionary
        columns_to_select = [
            'id', 'title', 'overview', 'poster_path', 'vote_average', 'popularity', 'tagline',
            'genres', 'backdrop_path', 'runtime', 'release_date',
            'production_companies', 'production_countries', 'spoken_languages'
        ]
        details = movie_row.iloc[0][columns_to_select].to_dict()
        # Process the details to clean data and add mock trailer URLs
        details = _process_movie_data(details)

        return jsonify({'details': details}), 200
    else:
        return jsonify({'message': 'Movie details not found.'}), 404

# Endpoint for autocomplete suggestions
@app.route('/movies', methods=['GET'])
def get_movie_titles():
    """
    API endpoint to provide movie title suggestions for autocomplete.
    Accepts an optional 'q' query parameter for filtering.
    """
    if df is None:
        return jsonify({'error': 'Models not loaded. Cannot fetch movie titles.'}), 500

    query = request.args.get('q', '').lower()
    if query:
        # Filter titles that contain the query string
        # Using .str.contains with case=False for case-insensitive search
        filtered_titles = df[df['title_lower'].str.contains(query, na=False)]['title'].tolist()
    else:
        # Return a small sample if no query, or all if dataset is small
        # Ensure titles are unique for suggestions
        filtered_titles = df['title'].drop_duplicates().head(50).tolist() # Limit for performance

    return jsonify({'titles': filtered_titles}), 200

# Endpoint for a featured movie banner
@app.route('/featured_movie', methods=['GET'])
def get_featured_movie():
    """
    API endpoint to get details for a randomly selected popular movie
    to be used as a banner.
    """
    if df is None:
        return jsonify({'error': 'Models not loaded. Cannot fetch featured movie.'}), 500

    # Filter for movies with a backdrop path and high popularity
    # Ensure 'backdrop_path' is not null before sampling
    eligible_movies = df[(df['backdrop_path'].notna()) & (df['popularity'] > 50)]

    if eligible_movies.empty:
        # Fallback if no popular movies with backdrops are found
        eligible_movies = df[df['backdrop_path'].notna()]
        if eligible_movies.empty:
            return jsonify({'message': 'No movies with backdrop paths available.'}), 404

    # Select a random movie from the eligible ones
    featured_movie = eligible_movies.sample(n=1).iloc[0]

    details = featured_movie[['title', 'overview', 'backdrop_path', 'tagline']].to_dict()
    # Process the featured movie details (e.g., for future trailer links in banner)
    details = _process_movie_data(details)
    return jsonify({'featured_movie': details}), 200

# Endpoint for fetching popular movies for the "Movies" page
@app.route('/movies/popular', methods=['GET'])
def get_popular_movies():
    """
    API endpoint to get a list of popular movies.
    """
    if df is None:
        return jsonify({'error': 'Models not loaded. Cannot fetch popular movies.'}), 500

    # Sort by popularity and select top N movies
    # Ensure 'poster_path' is not null for display
    popular_movies_df = df[df['poster_path'].notna()].sort_values(by='popularity', ascending=False).head(20) # Get top 20 popular movies

    columns_to_select = [
        'id', 'title', 'vote_average', 'popularity', 'overview', 'poster_path', 'tagline',
        'genres', 'runtime', 'release_date', 'production_companies',
        'production_countries', 'spoken_languages'
    ]
    popular_movies = popular_movies_df[columns_to_select].to_dict(orient='records')

    # Process each movie to clean data and add mock trailer URLs
    processed_movies = [_process_movie_data(movie) for movie in popular_movies]
    return jsonify({'movies': processed_movies}), 200


# Run the Flask app
if __name__ == '__main__':
    load_models() # Load models when the app starts
    # You can specify host='0.0.0.0' to make it accessible from other devices on your network
    # and port=5000 (default Flask port)
    app.run(debug=True, host='0.0.0.0', port=5000)