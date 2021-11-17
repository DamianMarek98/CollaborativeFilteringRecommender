import math

import numpy as np
from surprise import Dataset
from scipy.spatial import distance


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
    result_matrix = []
    user_ratings = []
    user_averages = []
    # rx,i - avg(x)
    deviation_matrix = []

    # fixable params
    nearest_neighbours_num = 2

    def __init__(self):
        # user id | item id | rating | timestamp
        self.movielens = Dataset.load_builtin('ml-100k')
        self.read_movielens()
        self.prepare_matrixes()
        self.calculate_deviations()

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

    def prepare_matrixes(self):
        self.zeros_matrix = np.zeros([self.num_of_users, self.num_of_films])
        self.result_matrix = np.zeros([self.num_of_users, self.num_of_films])
        self.deviation_matrix = np.zeros([self.num_of_users, self.num_of_films])
        self.user_averages = np.zeros([self.num_of_users])
        for user_rating in self.user_ratings:
            self.zeros_matrix[int(user_rating.user_id) - 1][int(user_rating.item_id) - 1] = user_rating.rating
            self.result_matrix[int(user_rating.user_id) - 1][int(user_rating.item_id) - 1] = user_rating.rating

    def rmse(self, real, predicted):
        return math.sqrt(((np.array(real) - np.array(predicted)) ** 2).mean())

    def calculate_deviations(self):
        for user_index in range(0, self.num_of_users):
            avg = self.zeros_matrix[user_index][self.zeros_matrix[user_index].nonzero()[0]].mean()
            self.user_averages[user_index] = avg
            for film_index in range(0, self.num_of_films):
                if self.zeros_matrix[user_index][film_index] != 0:
                    self.deviation_matrix[user_index][film_index] = self.zeros_matrix[user_index][film_index] - avg

    def find_nearest_neighbours(self, user_index, film_index):
        user_film_ratings = self.zeros_matrix[user_index]
        neighbours = []

        for user_i in range(0, self.num_of_users):
            if user_i == user_index or self.zeros_matrix[user_i][film_index] == 0:
                continue

            choosen_user_ratings = []
            neighbour_user_ratings = []
            deviation_choosen_user_ratings = []
            deviation_neighbour_user_ratings = []

            for film_i in range(0, self.num_of_films):
                if film_i == film_index:
                    neighbour_user_ratings.append(self.zeros_matrix[user_i][film_index])
                    choosen_user_ratings.append(self.user_averages[user_index])
                    deviation_choosen_user_ratings.append(0)
                    deviation_neighbour_user_ratings.append(self.deviation_matrix[user_i][film_i])
                    continue

                if self.zeros_matrix[user_i][film_i] != 0 and user_film_ratings[film_i] != 0:
                    choosen_user_ratings.append(user_film_ratings[film_i])
                    neighbour_user_ratings.append(self.zeros_matrix[user_i][film_i])
                    deviation_choosen_user_ratings.append(self.deviation_matrix[user_index][film_i])
                    deviation_neighbour_user_ratings.append(self.deviation_matrix[user_i][film_i])

            user_x_ratings = np.array(choosen_user_ratings)
            user_y_ratings = np.array(neighbour_user_ratings)
            user_x_avg = user_x_ratings.mean()
            user_y_avg = user_y_ratings.mean()
            fx = lambda x: x - user_x_avg
            fy = lambda y: y - user_y_avg
            user_x_ratings_minus_avg = fx(user_x_ratings)
            user_y_ratings_minus_avg = fy(user_y_ratings)


            if len(choosen_user_ratings) != 0:
                # tuple (user from iteration index, distance: 1 - angle, choosen_user_ratings, neighbour_user_ratings)
                #sim = self.calculate_sim(choosen_user_ratings, neighbour_user_ratings)
                sim = 1 - distance.cosine(user_x_ratings_minus_avg, user_y_ratings_minus_avg)
                neighbours.append((user_i, sim, choosen_user_ratings, neighbour_user_ratings))

        result = 0
        if len(neighbours) == 0:  # no neighbours then we use
            result = self.user_averages[user_index]
        else:  # sort neighbours:
            neighbours.sort(key=lambda x: x[1])
            if len(neighbours) > self.nearest_neighbours_num:
                neighbours = neighbours[0:self.nearest_neighbours_num - 1]

            #for neighbour in neighbours:







    def calculate_sim(self, user_x_ratings, user_y_ratings):
        user_x_ratings = np.array(user_x_ratings)
        user_y_ratings = np.array(user_y_ratings)
        user_x_avg = user_x_ratings.mean()
        user_y_avg = user_y_ratings.mean()
        fx = lambda x: x - user_x_avg
        fy = lambda y: y - user_y_avg
        user_x_ratings_minus_avg = fx(user_x_ratings)
        user_y_ratings_minus_avg = fy(user_y_ratings)

        # licznik
        nominator = 0
        for i in range(0, len(user_x_ratings_minus_avg)):
            nominator += (user_x_ratings_minus_avg[i] * user_y_ratings_minus_avg[i])
        # mianownik
        denominator = 0
        sum_of_pow_x = 0
        sum_of_pow_y = 0
        for i in range(0, len(user_x_ratings_minus_avg)):
            sum_of_pow_x += (user_x_ratings_minus_avg[i] ** 2)
            sum_of_pow_y += (user_y_ratings_minus_avg[i] ** 2)

        denominator = (math.sqrt(sum_of_pow_x) * math.sqrt(sum_of_pow_y))
        return nominator / denominator



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cfr = ColaborativeFilteringRecommender()
    cfr.find_nearest_neighbours(2, 2)
    a = 1
