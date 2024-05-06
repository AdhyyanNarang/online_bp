from utils_kernel import *
import numpy as np
import scipy
import random
import ipdb
from utils_objective import compute_convex_modular_gain, compute_diversity_gain, compute_affinity_gain
from collections import defaultdict

class Environment:

    def __init__(self, T, sigma, genre_list, lambda_one, lambda_two, movie_sim, column_to_movie, ratings_mat, movies_df_s, meta_user_centers, user_labels, recommender, seed = 0, naive_submodular_flag = False):
        self.T = T
        self.sigma = sigma
        self.genre_list = genre_list
        self.lambda_one = lambda_one
        self.lambda_two = lambda_two
        self.movie_sim = movie_sim
        self.column_to_movie = column_to_movie
        self.ratings_mat = ratings_mat
        self.movies_df_s = movies_df_s
        self.meta_user_centers = meta_user_centers
        self.user_labels = user_labels
        self.recommender = recommender
        self.naive_submodular_flag = naive_submodular_flag
        self.seed = seed

    def run_simulation(self):
        num_meta_users = len(self.meta_user_centers)
        user_range, movie_range = self.ratings_mat.shape

        np.random.seed(self.seed)
        ui_lst = np.random.choice(range(num_meta_users), size = self.T)
        Si_dict = {}
        yi_hist = []
        sub_hist_dict, sup_hist_dict, mod_hist_dict = {}, {}, {}
        #user_to_genre_to_total,
        user_to_genre_to_count = {}


        for i in range(num_meta_users):
            user_to_genre_to_count[i] = defaultdict(lambda: 0)
            user_to_genre_to_total[i] = defaultdict(lambda: 0)
            sub_hist_dict[i] = []
            sup_hist_dict[i] = []
            mod_hist_dict[i] = []
            Si_dict[i] = []

        for i in range(self.T):
            ui = ui_lst[i]

            #Obtain relevant variables
            print("User identity at iteration " + str(i) + " is: " + str(ui))
            Si = Si_dict[ui]

            #Obtain recommendation from recommender
            x_i = self.recommender.get_next_item(ui)

            movie_id = self.column_to_movie[x_i]
            print(self.movies_df_s[self.movies_df_s["movieId"] == movie_id])
            print("\n")
            Si_dict[ui].append(x_i)

            #This online computation helps speed up reward evaluation

            meta_user_idx = (self.user_labels == ui)
            ratings_mat_truncated = self.ratings_mat[meta_user_idx]
            movie_average = np.average(ratings_mat_truncated[:, x_i])
            modular_gain = movie_average
            sub_gain, user_to_genre_to_count = compute_diversity_gain(ui, x_i, user_to_genre_to_count, self.column_to_movie, self.movies_df_s, self.ratings_mat, self.user_labels)
            sup_gain, user_to_genre_to_total = compute_convex_modular_gain(ui, x_i, user_to_genre_to_total, self.column_to_movie, self.movies_df_s, self.ratings_mat, self.user_labels)

            if len(sub_hist_dict[ui]) > 0:
                sub = sub_hist_dict[ui][-1] + self.lambda_two*sub_gain
                sup =  sup_hist_dict[ui][-1] + self.lambda_one*sup_gain
                mod = mod_hist_dict[ui][-1] + modular_gain
            else :
                sub, sup, mod = self.lambda_two*sub_gain, self.lambda_one*sup_gain, modular_gain


            #Get reward
            y_i = self.lambda_two*sub_gain + modular_gain + self.lambda_one*sup_gain + self.sigma*np.random.normal()
            if not self.naive_submodular_flag:
                success = self.recommender.receive_reward(y_i)
            else :
                shadow_yi = sub_gain + self.sigma*np.random.normal()
                success = self.recommender.receive_reward(shadow_yi)


            #Bookkeeping
            sub_hist_dict[ui].append(sub)
            sup_hist_dict[ui].append(sup)
            mod_hist_dict[ui].append(mod)
            yi_hist.append(y_i)

        return Si_dict, ui_lst, yi_hist, sub_hist_dict, sup_hist_dict, mod_hist_dict
