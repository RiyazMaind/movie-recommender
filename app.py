import pandas as pd
import pickle
from scipy import sparse
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import random 
import os

app = Flask(__name__) 
CORS(app) 

df = None
tfidf = None
tfidf_matrix = None
cosine_sim = None
indices = None
models_initialized = False 

def load_models():
    global df, tfidf, tfidf_matrix, cosine_sim, indices, models_initialized
    try:
        base_path = os.path.dirname(__file__) 

        df_path = os.path.join(base_path, 'movies_df.pkl')
        tfidf_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
        tfidf_matrix_path = os.path.join(base_path, 'tfidf_matrix.npz')
        cosine_sim_path = os.path.join(base_path, 'cosine_sim.npy')

        print("--- Attempting to Load Models ---")
        print(f"Checking if files exist at: {base_path}")
        for f_name in ['movies_df.pkl', 'tfidf_vectorizer.pkl', 'tfidf_matrix.npz', 'cosine_sim.npy']:
            full_path = os.path.join(base_path, f_name)
            if not os.path.exists(full_path):
                print(f"WARNING: File not found: {full_path}")
            else:
                print(f"SUCCESS: File found: {full_path}")

        print(f"Attempting to load DataFrame from: {df_path}")
        df = pd.read_pickle(df_path)
        df['title_lower'] = df['title'].str.lower()
        indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()
        print("DataFrame loaded successfully.")

        print(f"Attempting to load TF-IDF vectorizer from: {tfidf_path}")
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        print("TF-IDF vectorizer loaded successfully.")

        print(f"Attempting to load TF-IDF matrix from: {tfidf_matrix_path}")
        tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
        print("TF-IDF matrix loaded successfully.")

        print(f"Attempting to load cosine similarity matrix from: {cosine_sim_path}")
        cosine_sim = np.load(cosine_sim_path)
        print("Cosine similarity matrix loaded successfully.")

        models_initialized = True
        print("--- All models and data loaded successfully! Initialized flag set to True. ---")

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: One or more model files not found. Please ensure they are in the correct directory. Error: {e}")
        models_initialized = False
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred during model loading. This might be due to memory limits or version incompatibility. Error: {e}")
        models_initialized = False

def _process_movie_data(movie_data):
    
    if isinstance(movie_data.get('genres'), str):
        movie_data['genres'] = [g.strip() for g in movie_data['genres'].split(',')]
    else:
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

    title_lower = movie_data.get('title', '').lower()
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
    return movie_data


def recommend(title, k=5):
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
    sim_scores = sim_scores[1:k+1]

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

@app.route('/recommend', methods=['POST'])
def get_movie_recommendations():
    if not models_initialized:
        return jsonify({'error': 'Server models not ready. Please try again in a moment.'}), 503

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


load_models()

if __name__ == '__main__':

    app.run(debug=True, host='127.0.0.1', port=5000)