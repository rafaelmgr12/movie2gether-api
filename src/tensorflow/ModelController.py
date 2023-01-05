import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras

from RecommederNet import RecommenderNet


class ModelController:

    def __init__(self, path_ratings: str):
        self.path_ratings = path_ratings

    def process_data(self):
        df = pd.read_csv(self.path_ratings)
        user_ids = df["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}

        movie_ids = df["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        df["user"] = df["userId"].map(user2user_encoded)
        df["movie"] = df["movieId"].map(movie2movie_encoded)

        num_users = len(user2user_encoded)
        num_movies = len(movie_encoded2movie)
        df["rating"] = df["rating"].values.astype(np.float32)
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(df["rating"])
        max_rating = max(df["rating"])

        print(
            "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
                num_users, num_movies, min_rating, max_rating
            )
        )

        df = df.sample(frac=1, random_state=42)
        x = df[["user", "movie"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int(0.9 * df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )

        return x_train, x_val, y_train, y_val, num_users, num_movies

    def process_data_test(self):
        df = pd.read_csv(self.path_ratings)
        user_ids = df["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}

        movie_ids = df["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

        return movie2movie_encoded, movie_encoded2movie

    @staticmethod
    def compile_model(num_users: int, num_movies: int, embedding_size: int, loss: tf.keras.losses,
                      optimizer: keras.optimizers):
        model = RecommenderNet(num_users, num_movies, embedding_size)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    @staticmethod
    def train_model(model, x_train, y_train, batch_size: int, epochs: int, validation_data: tuple):
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=validation_data,
        )
        return history
