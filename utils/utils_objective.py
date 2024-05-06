import numpy as np
from collections import defaultdict
import copy

def find_genres(movie, column_to_movie, movies_df_s):
    movie_id = column_to_movie[movie]
    genres_string = movies_df_s[movies_df_s["movieId"] == movie_id]["genres"].tolist()[0]
    return genres_string.split("|")

def find_num_intersect(genre, S, column_to_movie, movies_df_s):
    total = 0
    for movie in S:
        movie_genres = find_genres(movie, column_to_movie, movies_df_s)
        if genre in movie_genres:
            total += 1
    return total

def sum_of_convex_modular(S, genre_list, column_to_movie, ratings_mat, movies_df_s):
    cvx = lambda x: x**2
    genre_to_total = defaultdict(lambda: 0)

    for movie in S:
        movie_genres = find_genres(movie, column_to_movie, movies_df_s)
        movie_average = np.average(ratings_mat[:, movie])
        chosen_genre = movie_genres[0]

        for genre in movie_genres:
            if genre in genre_to_total and genre_to_total[genre] > genre_to_total[chosen_genre]:
                chosen_genre = genre

        genre_to_total[chosen_genre] += movie_average

    ret_val = 0

    for (genre, total) in genre_to_total.items():
        ret_val += cvx(total)

    return ret_val

"""
Online reward evaluation
"""
def compute_bp_gainV2(ui, x_i, user_to_genre_to_count, selected_for_user, affinity_mat, column_to_movie, movies_df_s, ratings_mat, user_labels, lambda_one, lambda_two):
    sub_gain, user_to_genre_to_countV2 = compute_diversity_gain(ui, x_i, user_to_genre_to_count, column_to_movie, movies_df_s, ratings_mat, user_labels)
    sup_gain = compute_affinity_gain(selected_for_user, x_i, affinity_mat)

    meta_user_idx = (user_labels == ui)
    ratings_mat_truncated = ratings_mat[meta_user_idx]
    movie_average = np.average(ratings_mat_truncated[:, x_i])

    modular_gain = movie_average

    return modular_gain + lambda_two*sub_gain + lambda_one*sup_gain, user_to_genre_to_countV2

def compute_bp_gain(ui, x_i, user_to_genre_to_count, user_to_genre_to_total, column_to_movie, movies_df_s, ratings_mat, user_labels, lambda_one):
    sub_gain, user_to_genre_to_countV2 = compute_diversity_gain(ui, x_i, user_to_genre_to_count, column_to_movie, movies_df_s, ratings_mat, user_labels)

    sup_gain, user_to_genre_to_totalV2 = compute_convex_modular_gain(ui, x_i, user_to_genre_to_total, column_to_movie, movies_df_s, ratings_mat, user_labels)

    return sub_gain + lambda_one*sup_gain, user_to_genre_to_countV2, user_to_genre_to_totalV2

def compute_diversity_gain(ui, x_i, user_to_genre_to_count, column_to_movie, movies_df_s, ratings_mat, user_labels, tau = 3.6):
    user_to_genre_to_count = copy.deepcopy(user_to_genre_to_count)
    h = lambda x: np.sqrt(1 + x)
    genres_curr = find_genres(x_i, column_to_movie, movies_df_s)
    num_genres = len(genres_curr)
    genre_to_count = user_to_genre_to_count[ui]
    gain = 0

    meta_user_idx = (user_labels == ui)
    ratings_mat_truncated = ratings_mat[meta_user_idx]
    movie_average = np.average(ratings_mat_truncated[:, x_i])

    if movie_average < tau:
        return 0, user_to_genre_to_count

    for genre in genres_curr:
        old_count = genre_to_count[genre]
        genre_to_count[genre] += 1/num_genres
        gain += h(genre_to_count[genre]) - h(old_count)

    user_to_genre_to_count[ui] = genre_to_count

    return gain, user_to_genre_to_count

def compute_convex_modular_gain(ui, x_i, user_to_genre_to_total, column_to_movie, movies_df_s, ratings_mat, user_labels):
    user_to_genre_to_total = copy.deepcopy(user_to_genre_to_total)
    cvx = lambda x: x**2
    genres_curr = find_genres(x_i, column_to_movie, movies_df_s)
    genre_to_total = user_to_genre_to_total[ui]
    num_genres = len(genres_curr)

    meta_user_idx = (user_labels == ui)
    ratings_mat_truncated = ratings_mat[meta_user_idx]

    movie_average = np.average(ratings_mat_truncated[:, x_i])
    gain = 0

    for genre in genres_curr:
        old_total = genre_to_total[genre]
        genre_to_total[genre] += movie_average/num_genres
        gain += cvx(genre_to_total[genre]) - cvx(old_total)

    user_to_genre_to_total[ui] = genre_to_total
    return gain, user_to_genre_to_total

