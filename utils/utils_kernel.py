import numpy as np
import ipdb
from sklearn.gaussian_process.kernels import RBF

reg_factor = 0.2
"""
Consider a binary vector representation for each movie, where entry is 1
if the genre is present in the movie.

Then the function below finds the angle between the vector reps for two movies.
"""
def cosine_kernel(first_movie, second_movie, movie_sim):
    #first_movie, second_movie = first_movie + 1, second_movie + 1
    scale = 1 - reg_factor
    original = movie_sim[first_movie, second_movie]
    return original

def calc_cos_angle(first_movie, second_movie, column_to_movie, movies_df_s):

    #Transforms to 1-indexing that the dataset uses
    first_movie_id, second_movie_id = column_to_movie[first_movie], column_to_movie[second_movie]

    #Some movieIds (eg 91) are not present in the datamatrix
    try:
        first_genres_string = movies_df_s[movies_df_s["movieId"] == first_movie_id]["genres"].tolist()[0]
        second_genres_string = movies_df_s[movies_df_s["movieId"] == second_movie_id]["genres"].tolist()[0]
    except:
        if first_movie_id == second_movie_id:
            return 1
        else:
            return 0

    first_genres_set = set(first_genres_string.split("|"))
    second_genres_set = set(second_genres_string.split("|"))

    dot_product = len(first_genres_set & second_genres_set)

    return dot_product/(np.sqrt(len(first_genres_set)) * np.sqrt(len(second_genres_set)))

def jaccard_kernel(first_history, second_history, movie_sim):

    #if both histories have size 0, return 1
    if len(first_history) == 0 and len(second_history) == 0:
        return 1.0

    if len(first_history) == 0 or len(second_history) == 0:
        return 0.0

    first_set = set(first_history)
    second_set = set(second_history)
    intersect = first_set & second_set
    union = first_set.union(second_set)
    original = float(len(intersect))/float(len(union))

    return original

def cosine_kernel_meta_users(meta_user_one, meta_user_two, meta_user_centers):
    scale = 1 - reg_factor
    user_one_features = meta_user_centers[meta_user_one]
    user_two_features = meta_user_centers[meta_user_two]
    original = user_one_features.dot(user_two_features)/(np.linalg.norm(user_one_features)*np.linalg.norm(user_two_features))
    return original

def composite_kernel(first_movie, second_movie, first_history, second_history, movie_sim, k1, k2):
    cosine_output = cosine_kernel(first_movie, second_movie, movie_sim)
    jac_output = jaccard_kernel(first_history, second_history, movie_sim)
    return (cosine_output, jac_output, k1*cosine_output  + k2*jac_output)

def composite_kernel_with_user(first_movie, second_movie, first_history, second_history, meta_user_one, meta_user_two, movie_sim, meta_user_centers, k1, k2, k3):
    cosine_output = cosine_kernel(first_movie, second_movie, movie_sim)
    jac_output = jaccard_kernel(first_history, second_history, movie_sim)
    user_output = cosine_kernel_meta_users(meta_user_one, meta_user_two, meta_user_centers)
    original = k1*cosine_output  + k2*jac_output + k3*user_output
    scale = 1 - reg_factor

    if original >= 1 - 1e-3:
        return original
    else:
        return scale * original

def composite_kernel_with_user_sep(first_movie, second_movie, first_history, second_history, meta_user_one, meta_user_two, movie_sim, meta_user_centers, k1, k2, k3, k4, oi, oj):
    cosine_output = cosine_kernel(first_movie, second_movie, movie_sim)
    jac_output = jaccard_kernel(first_history, second_history, movie_sim)
    user_output = cosine_kernel_meta_users(meta_user_one, meta_user_two, meta_user_centers)
    oj_kernel = 0
    #Calulate the RBF Kernel value between oi, oj
    if oi == oj:
        oj_kernel = 1
    else:
        oj_kernel = np.exp(-1 * (np.linalg.norm(oi - oj)**2) / (2 * (1**2)))


    original = k1*cosine_output  + k2*jac_output + k3*user_output + k4*oj_kernel
    scale = 1 - reg_factor

    if original >= 1 - 1e-3:
        return original
    else:
        return scale * original
