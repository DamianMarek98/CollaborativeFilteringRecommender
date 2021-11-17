import math

import numpy as np
from surprise import Dataset


class UserRating:
    user_id = ""
    item_id = ""
    rating = None

    def __init__(self, user_id, item_id, rating):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating


class ColaborativeFilteringRecommender:
    num_of_users = 0
    num_of_films = 0
    movielens = None
    zeros_matrix = []
    user_ratings = []

    def __init__(self):
        # user id | item id | rating | timestamp
        self.movielens = Dataset.load_builtin('ml-100k')
        self.read_movielens()
        self.prepare_matrix()

    def read_movielens(self):
        for user_rating in self.movielens.raw_ratings:
            user_id = user_rating[0]
            item_id = user_rating[1]
            rating = user_rating[2]
            self.user_ratings.append(UserRating(user_id, item_id, rating))
            if int(user_id) > self.num_of_users:
                self.num_of_users = int(user_id)

            if int(item_id) > self.num_of_films:
                self.num_of_films = int(item_id)

    def prepare_matrix(self):
        self.zeros_matrix = np.zeros([self.num_of_users, self.num_of_films])
        for user_rating in self.user_ratings:
            self.zeros_matrix[int(user_rating.user_id) - 1][int(user_rating.item_id) - 1] = user_rating.rating

    def rmse(self, real, predicted):
        return math.sqrt(((np.array(real) - np.array(predicted)) ** 2).mean())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cfr = ColaborativeFilteringRecommender()
