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
    naive_result_matrix = []
    user_ratings = []
    user_averages = []
    film_averages = []
    # rx,i - avg(x)
    deviation_matrix = []
    # fixable params
    nearest_neighbours_num = 2
    cross_validation_user_indexes = [0, 5, 123, 256, 512]
    hold_out = 100 # first 100 users as test set

    def __init__(self):
        # user id | item id | rating | timestamp
        self.movielens = Dataset.load_builtin('ml-100k')
        self.read_movielens()
        self.prepare_matrixes()
        self.calculate_deviations()
        # prediction
        self.person_corelation_coefficient_prediction()
        # naive prediction
        self.naive_prediction()
        # mae
        # rmse
        self.check()

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
        self.naive_result_matrix = np.zeros([self.num_of_users, self.num_of_films])
        self.deviation_matrix = np.zeros([self.num_of_users, self.num_of_films])
        self.user_averages = np.zeros([self.num_of_users])
        self.film_averages = np.zeros([self.num_of_films])
        for user_rating in self.user_ratings:
            self.zeros_matrix[int(user_rating.user_id) - 1][int(user_rating.item_id) - 1] = user_rating.rating
            self.result_matrix[int(user_rating.user_id) - 1][int(user_rating.item_id) - 1] = user_rating.rating
            self.naive_result_matrix[int(user_rating.user_id) - 1][int(user_rating.item_id) - 1] = user_rating.rating

    def calculate_deviations(self):
        for user_index in range(0, self.num_of_users):
            avg = self.zeros_matrix[user_index][self.zeros_matrix[user_index].nonzero()[0]].mean()
            self.user_averages[user_index] = avg
            for film_index in range(0, self.num_of_films):
                if self.zeros_matrix[user_index][film_index] != 0:
                    self.deviation_matrix[user_index][film_index] = self.zeros_matrix[user_index][film_index] - avg

        for film_index in range(0, self.num_of_films):
            avg = 0
            num_of_ratings = 0
            for user_index in range(0, self.num_of_users):
                if self.zeros_matrix[user_index][film_index] != 0:
                    avg += self.zeros_matrix[user_index][film_index]
                    num_of_ratings += 1
            self.film_averages[film_index] = (avg / num_of_ratings)

    def predict_from_nearest_neighbours(self, user_index, film_index):
        user_film_ratings = self.zeros_matrix[user_index]
        neighbours = []

        for user_i in range(self.hold_out, self.num_of_users): # first 100 users as test set - learning on others
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
                # sim i sim2 działają
                sim = self.calculate_sim(choosen_user_ratings, neighbour_user_ratings)
                cosine = distance.cosine(user_x_ratings_minus_avg, user_y_ratings_minus_avg)
                sim2 = 1 - cosine
                # if sim != 0:  # sim = 0 better to tak avg
                # tuple (user from iteration index, sim)
                neighbours.append((user_i, sim))

        if len(neighbours) == 0:  # no neighbours then we use
            pred = self.user_averages[user_index]
        else:  # sort neighbours:
            neighbours.sort(key=lambda x: x[1], reverse=True)
            if len(neighbours) > self.nearest_neighbours_num:
                neighbours = neighbours[0:self.nearest_neighbours_num]

            # prediction slajd 8
            pred = self.user_averages[user_index]
            nominator = 0
            denominator = 0
            for neighbour in neighbours:
                nominator += (neighbour[1] * self.deviation_matrix[neighbour[0]][
                    film_index])  # sum sim(x,y) * ry,i - avg ry
                denominator += neighbour[1]  # sum sim(x,y)

            pred += (nominator / denominator)
            if pred < 0 or pred > 5:
                return self.film_averages[film_index]
        return pred

    def person_corelation_coefficient_prediction(self):
        for user_index in range(0, self.hold_out): # first 100 users as test set
            for film_index in range(0, self.num_of_films):
                if self.zeros_matrix[user_index][film_index] != 0:
                    pred = self.predict_from_nearest_neighbours(user_index, film_index)
                    self.result_matrix[user_index][film_index] = pred

    def naive_prediction(self):
        for user_index in range(0, self.hold_out): # first 100 users as test set
            for film_index in range(0, self.num_of_films):
                if self.zeros_matrix[user_index][film_index] != 0:
                    self.naive_result_matrix[user_index][film_index] = self.predict_naive_for_user(user_index,
                                                                                                   film_index)

    def predict_naive_for_user(self, selected_user_index, film_index):
        avg = 0
        num_of_ratings = 0
        for user_index in range(self.hold_out, self.num_of_users): # first 100 users as test set - learning on others
            if user_index == selected_user_index:
                continue
            if self.zeros_matrix[user_index][film_index] != 0:
                avg += self.zeros_matrix[user_index][film_index]
                num_of_ratings += 1

        if num_of_ratings == 0:
            return self.zeros_matrix[selected_user_index][film_index]

        return avg / num_of_ratings

    def check(self):
        all_predicted_known_values = []
        all_known_values = []
        all_predicted_naive_known_values = []
        for user_index in range(0, self.hold_out): # first 100 users as test set
            predicted_known_values = []
            known_values = []
            predicted_naive_known_values = []
            for film_index in range(0, self.num_of_films):
                if self.zeros_matrix[user_index][film_index] != 0:
                    if math.isnan(self.result_matrix[user_index][film_index]) or math.isnan(self.naive_result_matrix[user_index][film_index]):
                        continue #skip nan values
                    known_values.append(self.zeros_matrix[user_index][film_index])
                    predicted_known_values.append(self.result_matrix[user_index][film_index])
                    predicted_naive_known_values.append(self.naive_result_matrix[user_index][film_index])
                    all_known_values.append(self.zeros_matrix[user_index][film_index])
                    all_predicted_known_values.append(self.result_matrix[user_index][film_index])
                    all_predicted_naive_known_values.append(self.naive_result_matrix[user_index][film_index])

            mae_real_naive = self.mae(known_values, predicted_naive_known_values)
            mae_real_pred = self.mae(known_values, predicted_known_values)
            rsme_real_pred = self.rmse(known_values, predicted_known_values)
            rsme_real_naive = self.rmse(known_values, predicted_naive_known_values)
            print("Hold on - test user results: " + str(user_index))
            print("rsme naive against real: " + str(rsme_real_naive))
            print("rmse predicted against real: " + str(rsme_real_pred))
            print("mae naive against real: " + str(mae_real_naive))
            print("mae predicted against real: " + str(mae_real_pred))
            print("-------------------------------------------------------")

        #summary results:
        mae_real_naive = self.mae(all_known_values, all_predicted_naive_known_values)
        mae_real_pred = self.mae(all_known_values, all_predicted_known_values)
        rsme_real_pred = self.rmse(all_known_values, all_predicted_known_values)
        rsme_real_naive = self.rmse(all_known_values, all_predicted_naive_known_values)
        print("Hold on - full set results: ")
        print("rsme naive against real: " + str(rsme_real_naive))
        print("rmse predicted against real: " + str(rsme_real_pred))
        print("mae naive against real: " + str(mae_real_naive))
        print("mae predicted against real: " + str(mae_real_pred))
        print("-------------------------------------------------------")

    def rmse(self, values, predicted):
        iterator = len(values)
        res = 0
        for i in range(0, iterator):
            res += pow((predicted[i] - values[i]), 2)

        return np.sqrt(res / iterator)

    def mae(self, values, predicted):
        iterator = len(values)
        res = 0
        for i in range(0, iterator):
            res += abs(predicted[i] - values[i])

        return res / iterator

    # not used - implemented while checking results
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
    a = 10
