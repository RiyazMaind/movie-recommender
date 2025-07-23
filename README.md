# Personalized Movie Recommendation System
A full-stack web application that provides personalized movie recommendations based on content similarity, inspired by the sleek user experience of modern streaming platforms like Disney+ Hotstar.

## Features
Dynamic Hero Banner: A rotating banner showcasing popular movies with their titles and taglines, fetched directly from the dataset.

Intelligent Search with Autocomplete: Easily find movies with real-time suggestions as you type.

Detailed Movie Information: Get comprehensive details for selected movies, including poster, overview, ratings, popularity, genres, runtime, and production info.

Personalized Recommendations: Discover new films based on the content of movies you already enjoy.

Interactive Movie Cards: Engaging hover effects on recommendation cards to reveal more details.

Responsive UI: Seamless experience across desktop and mobile devices.

Navigation: Dedicated sections for "Home", "Popular Movies", and a placeholder for "TV Shows".


## Screenshots
Dynamic Hero Banner
https://drive.google.com/file/d/16pC9qVJs9EWCGatrzV9o5epOakqKOcGv/view?usp=sharing

Selected Movie Details
https://drive.google.com/file/d/1XQF0gs0Zsxvel9gagPfNtOSiFZtdB5vq/view?usp=sharing

Personalized Recommendations
https://drive.google.com/file/d/1Ex4tFFTFZhdmAtwoX_H1ga0l2zMX9gt9/view?usp=sharing

Popular Movies Section
https://drive.google.com/file/d/1BKCVoKl1a9dOqTTrjMCagfVkP4J7prbU/view?usp=sharing

## Technologies Used
Frontend:

HTML5: Structure of the web pages.

CSS (Tailwind CSS): Utility-first CSS framework for rapid and responsive styling.

JavaScript (ES6+): For interactive elements, API calls, and DOM manipulation.

Backend:

Python: Core programming language.

Flask: Lightweight web framework for building the RESTful API.

Pandas: Data manipulation and analysis for movie dataset.

Scikit-learn: For TF-IDF Vectorization and Cosine Similarity calculation.

SciPy: For sparse matrix operations.

NumPy: Numerical computing.

Machine Learning:

Content-Based Filtering: Recommends items similar to those a user has liked in the past.

TF-IDF (Term Frequency-Inverse Document Frequency): Converts text data (movie overviews, genres, keywords, taglines) into numerical representations.

Cosine Similarity: Measures the similarity between two non-zero vectors in an inner product space, used to find similar movies.

## Getting Started (Local Setup)
To run this project locally, follow these steps:

Prerequisites
Python 3.8+

pip (Python package installer)

Git

Git LFS

1. Clone the Repository
git clone https://github.com/RiyazMaind/movie-recommender.git
cd movie-recommender # Navigate to your backend directory

2. Install Git LFS
Ensure Git LFS is installed and configured for your repository:

git lfs install
git lfs pull # Pull large files that Git LFS tracks

3. Install Backend Dependencies
It's highly recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

4. Prepare Model Files
Ensure your pre-trained model files (movies_df.pkl, tfidf_vectorizer.pkl, tfidf_matrix.npz, cosine_sim.npy) are present in the root directory of the movie-recommender folder. These files are managed by Git LFS.

5. Run the Flask Backend
python app.py

The backend API will start running on http://127.0.0.1:5000.

6. Open the Frontend
Open the index.html file in your web browser. It is configured to communicate with the local Flask backend.
