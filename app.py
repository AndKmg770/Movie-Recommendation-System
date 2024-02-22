from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

mv_data = pd.read_csv('mv_yr.csv', nrows=10000)
rating = pd.read_csv('ratings.csv', nrows=10000)
tag = pd.read_csv('tags.csv', nrows=10000)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    search = request.form.get('searchType')
################################################################################
    if search == 'Movie Name':
        cv = CountVectorizer()
        genres_matrix = cv.fit_transform(mv_data['genres'].values.astype('U'))
        cosine_sim = cosine_similarity(genres_matrix)

        def get_similar_movies(title, cosine_sim, mv_data):
            index = mv_data[mv_data['title'] == title].index[0]
            sim_scores = list(enumerate(cosine_sim[index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            recommender = [mv_data.iloc[score[0]]['title'] for score in sim_scores[0:10]]
            return recommender

        recommender = get_similar_movies(title, cosine_sim, mv_data)

        rec_movies = []

        for movie in recommender:
            rec_movies.append(movie)

        return render_template('recommendations.html', title=title, movies=rec_movies)
    ############################################################################
    elif search == 'Movie_Ratings':
        merged_df = pd.merge(rating, mv_data, on='movieId')
        user_rating_matrix = merged_df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
        cosine_sim = cosine_similarity(user_rating_matrix)

        def get_similar_movies(movie_title, cosine_sim, movies_data):
            try:
                index = movies_data[movies_data['title'] == movie_title].index[0]
                sim_scores = list(enumerate(cosine_sim[index]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                similar_movie_titles = [movies_data.iloc[score[0]]['title'] for score in sim_scores[1:11]]
                return similar_movie_titles
            except IndexError:
                return []

        recommended_movies = get_similar_movies(title, cosine_sim, mv_data)

        return render_template('recommendations.html', title=title, movies=recommended_movies)
    ############################################################################
    elif search == 'Movie Tags':
        merged_df = pd.merge(tag, mv_data, on='movieId')
        cv = CountVectorizer()
        tag_matrix = cv.fit_transform(merged_df['tag'].values.astype('U'))
        cosine_sim = cosine_similarity(tag_matrix)

        def get_similar_movies(tag, cosine_sim, movies_data):
            try:
                index = merged_df[merged_df['tag'] == tag].index[0]
                sim_scores = list(enumerate(cosine_sim[index]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                movie_titles = [movies_data[movies_data['movieId'] == merged_df.iloc[score[0]]['movieId']]['title'].values[0] for score in sim_scores[1:11]]
                return movie_titles
            except IndexError:
                return []

        recommended_movies = get_similar_movies(title, cosine_sim, mv_data)

        return render_template('recommendations.html', title=title, movies=recommended_movies)
    ############################################################################
    elif search == 'Ratings to Movie':
        merged_df = pd.merge(rating, mv_data, on='movieId')
        movie_rating_matrix = merged_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
        cosine_sim = cosine_similarity(movie_rating_matrix)

        def get_similar_movies(rating_input, cosine_sim, movies_data):
            try:
                index = merged_df[merged_df['rating'] == rating_input].index[0]
                sim_scores = list(enumerate(cosine_sim[index]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                similar_movie_titles = [movies_data.iloc[score[0]]['title'] for score in sim_scores[1:11]]
                return similar_movie_titles
            except IndexError:
                return []

        recommended_movies = get_similar_movies(float(title), cosine_sim, mv_data)

        return render_template('recommendations.html', title=title, movies=recommended_movies)
    else:
        title = "Not Found Entered Values Not on DataSet"
        return render_template('recommendations.html', title=title)

if __name__ == '__main__':
    app.run()
