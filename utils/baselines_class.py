from utils_kernel import *
import numpy as np
import scipy
import random
import ipdb
import copy
from utils_objective import compute_bp_gainV2
from collections import defaultdict

class GreedyBaseline:

    def __init__(self, ground_set, column_to_movie, movies_df_s, ratings_mat, affinity_mat, user_labels, lambda_one, lambda_two, num_meta_users):

        #Information about the ground set
        self.ground_set = ground_set
        self.column_to_movie = column_to_movie
        self.movies_df_s = movies_df_s
        self.ratings_mat = ratings_mat
        self.user_labels = user_labels
        self.lambda_one = lambda_one
        self.lambda_two = lambda_two
        self.affinity_mat = affinity_mat

        #Information about the different BP functions that will arrive
        self.num_meta_users = num_meta_users

        #Record what is happening in the game
        self.ui_list = []
        self.yi_list = []
        self.user_to_selected = {}
        self.user_to_genre_to_count = {}
        #self.user_to_genre_to_total = {}

        for i in range(self.num_meta_users):
            self.user_to_genre_to_count[i] = defaultdict(lambda : 0)
            #self.user_to_genre_to_total[i] = defaultdict(lambda : 0)
            self.user_to_selected[i] = []

    def get_next_item(self, ui):
        self.ui_list.append(ui)
        selected_for_user = self.user_to_selected[ui]

        potential_gains = []

        for item in self.ground_set:
            if item in selected_for_user:
                potential_gains.append(0)
                continue
            else :
                gain, dict_one = compute_bp_gainV2(ui, item, self.user_to_genre_to_count, selected_for_user, self.affinity_mat, self.column_to_movie, self.movies_df_s, self.ratings_mat, self.user_labels, self.lambda_one, self.lambda_two)

                potential_gains.append(gain)
                del dict_one

        selected = np.argmax(potential_gains)
        self.user_to_selected[ui].append(selected)

        gain, dict_one = compute_bp_gainV2(ui, selected, self.user_to_genre_to_count, selected_for_user, self.affinity_mat, self.column_to_movie, self.movies_df_s, self.ratings_mat, self.user_labels, self.lambda_one, self.lambda_two)

        """
        gain, dict_one, dict_two = compute_bp_gain(ui, selected, self.user_to_genre_to_count, self.user_to_genre_to_total,
                                                   self.column_to_movie, self.movies_df_s, self.ratings_mat, self.user_labels, self.lambda_one)
        """

        self.user_to_genre_to_count = dict_one
        return selected

    def receive_reward(self, yi):
        self.yi_list.append(yi)
        return True

class RandomBaseline:

    def __init__(self, ground_set, genre_list, affinity_mat, column_to_movie, ratings_mat, movies_df_s, meta_user_centers, user_labels):
        #Information about the ground set
        self.ground_set = ground_set
        self.genre_list = genre_list
        self.affinity_mat = affinity_mat
        self.column_to_movie = column_to_movie
        self.ratings_mat = ratings_mat
        self.movies_df_s = movies_df_s

        #Information about the different BP functions that will arrive
        self.meta_user_centers = meta_user_centers
        self.num_meta_users = len(meta_user_centers)
        self.user_labels = user_labels

        #Record what is happening in the game
        self.ui_list = []
        self.yi_list = []
        self.user_to_selected = {}

        for i in range(self.num_meta_users):
            self.user_to_selected[i] = []

    def get_next_item(self, ui):
        self.ui_list.append(ui)
        selected = np.random.choice(self.ground_set, 1)[0]

        while selected in self.user_to_selected[ui]:
            selected = np.random.choice(self.ground_set, 1)[0]

        self.user_to_selected[ui].append(selected)
        return selected

    def receive_reward(self, yi):
        self.yi_list.append(yi)
        return True

