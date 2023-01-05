import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import os
import tempfile

from ModelController import ModelController

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
LOCAL_DIR = os.getcwd()

if __name__ == '__main__':
    movielens_data_file_url = (
        "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    )
    movielens_zipped_file = keras.utils.get_file(
        "ml-latest-small.zip", movielens_data_file_url, extract=False
    )
    keras_datasets_path = Path(movielens_zipped_file).parents[0]
    movielens_dir = keras_datasets_path / "ml-latest-small"

    # Only extract the data the first time the script is run.
    if not movielens_dir.exists():
        with ZipFile(movielens_zipped_file, "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=keras_datasets_path)
            print("Done!")

    ratings_file = movielens_dir / "ratings.csv"
    model_controller = ModelController(ratings_file)
    df = pd.read_csv(ratings_file)
    x_train, x_val, y_train, y_val, num_users, num_movies = model_controller.process_data()

    model = model_controller.compile_model(num_users, num_movies, 32, tf.keras.losses.BinaryCrossentropy(),
                                           keras.optimizers.Adam(learning_rate=0.001))
    history = model_controller.train_model(model, x_train, y_train, 64, 10, (x_val, y_val))
    print(model.summary())
    test_loss = model.evaluate(x_val, y_val)
    print('\nTest Loss: {}'.format(test_loss))

    movie2movie_encoded, movie_encoded2movie = model_controller.process_data_test()

    print("Testing Model with 1 user")
    movie_df = pd.read_csv(movielens_dir / "movies.csv")
    user_id = "new_user"
    movies_watched_by_user = df.sample(5)
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

    user_movie_array = np.hstack(
        ([[0]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (
        movies_watched_by_user.sort_values(by="rating", ascending=False)
        .head(5)
        .movieId.values
    )
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
    for row in movie_df_rows.itertuples():
        print(row.title, ":", row.genres)

    print("----" * 8)
    print("Top 10 movie recommendations")
    print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(row.title, ":", row.genres)

    print("===" * 9)
    print("Saving Model")
    print("===" * 9)

    MODEL_DIR = tempfile.gettempdir()
    version = 15
    export_path = os.path.join(LOCAL_DIR, f"models/{version}")

    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )