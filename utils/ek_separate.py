from utils_kernel import *
import numpy as np
import scipy
import random
import ipdb
import warnings
from utils_objective import compound_objective, compound_objective_multi, find_genres, qk_item, qk_set
from utils_linalg import sherman_morrison_update, schur_first_order_update, schur_first_order_update_fast

EPS = 0

class EK_BP_UCB_Sep:

    def __init__(self, ground_set, beta_lst, sigma, genre_list, k1, k2, k3, k4, movie_sim, column_to_movie, ratings_mat, movies_df_s, meta_user_centers, user_labels, reg_lambda, mu_kors, epsilon_kors, ui_lst_tot):
        #Note that some instantiations are performed after the first round of the game
        # in the function "instantiate_kernel_matrices"

        #Random seed -
        #self.rng = np.random.RandomState(123)

        # (Never altered) Information about the ground set
        self.ground_set = ground_set
        self.genre_list = genre_list
        self.movie_sim = movie_sim
        self.column_to_movie = column_to_movie
        self.movies_df_s = movies_df_s

        # (Never altered) For the kernel
        self.ratings_mat = ratings_mat
        self.k1, self.k2, self.k3, self.k4  = k1, k2, k3, k4

        #(Never altered) Information about the different BP functions that will arrive
        self.sigma = sigma
        self.meta_user_centers = meta_user_centers
        self.num_meta_users = len(meta_user_centers)
        self.user_labels = user_labels

        #(Never altered) Information for the Nystrom sketching
        self.reg_lambda= reg_lambda
        self.mu_kors = mu_kors
        self.epsilon_kors = epsilon_kors

        # (Never altered) Hyperparameters that balance between explore/exploit
        self.beta_lst = beta_lst

        # Book-keeping of user identity, item selection, reward at each stage.
        self.ui_list = []
        self.yi_list = []
        self.user_to_selected = {}


        # Latest kernel matrices that are updated in an online fashion at each stage
        self.G = []
        self.K_ZS = []
        self.KGG_inv = []
        self.Lambda = []
        self.y_tilde = []
        self.kss = []

        self.mu_i_dict = {}
        self.sigma_i_dict = {}

        #The below is a dictionary that maps from user to a dict of modular functions, one for each genre.
        self.user_to_qV_dict = {}
        self.Tui_dict = {}

        for i in range(self.num_meta_users):
            mu_i = np.array([0.0 for movie in ground_set])
            sigma_i = np.array([float(k1 + k2 + k3 + k4) for movie in ground_set])
            self.mu_i_dict[i] = mu_i
            self.sigma_i_dict[i] = sigma_i
            self.user_to_selected[i] = []

            qV = qk_set(self.ground_set, i, self.column_to_movie, self.movies_df_s, self.ratings_mat, self.user_labels, tau=3.6)
            self.user_to_qV_dict[i] = qV

            #Save the number of occurences of i in ui_lst into a variable
            Tui = np.count_nonzero(ui_lst_tot == i)
            self.Tui_dict[i] = Tui

    def mod_func(self, item):
        h = lambda x: np.sqrt(1 + x)

        #Calculate qk dict of the item
        ui = self.ui_list[-1]
        qk = qk_item(ui, item, self.column_to_movie, self.movies_df_s, self.ratings_mat, self.user_labels, tau=3.6)

        if qk == 0:
            return 0

        #Create a copy of self.user_to_qV[ui]
        genre_to_qV = self.user_to_qV_dict[ui]
        genre_to_qV_missing = self.user_to_qV_dict[ui].copy()

        #For the genres in qk, subtract qk[genre] from genre_to_qV[genre]
        for genre in qk:
            genre_to_qV_missing[genre] -= qk[genre]

        total = 0
        for genre in genre_to_qV:
            old_count = genre_to_qV_missing[genre]
            new_count = genre_to_qV[genre]
            total += h(new_count) - h(old_count)

        del genre_to_qV_missing
        return total

    def get_next_item(self,ui):
        #Record relevant variables
        i = len(self.ui_list)
        beta_i = self.beta_lst[i]
        mu_i, sigma_i = self.mu_i_dict[ui], self.sigma_i_dict[ui]
        Si, yi_hist  = self.user_to_selected[ui], self.yi_list

        if i == 0:
            x_i = np.random.choice(self.ground_set, 1)[0]
        else :
            #Obtain the relevant kernel matrices

            G, KGG_inv, Lambda, y_tilde = self.G, self.KGG_inv, self.Lambda, self.y_tilde

            for x in self.ground_set:
                K_Gx = self.kernel_evaluate(self.G, ui, x)

                #mean update
                if x in Si:
                    mu_i[x] = -1e6
                else:
                    try:
                        mu_i[x] =  K_Gx.T @ Lambda @ y_tilde
                    except:
                        ipdb.set_trace()

                Delta_i = K_Gx.T @ (Lambda - 1/self.reg_lambda* KGG_inv) @ K_Gx

                post_var = 1/self.reg_lambda* (self.k1 + self.k2 + self.k3) + Delta_i

                # Try calculation below, and set a breakpoint if there is an error
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        sigma_i[x] = np.sqrt(post_var)
                    except RuntimeWarning as e:
                        if post_var <= -100:
                            print("Posterior variance is:", post_var)

            x_i = np.argmax(mu_i + beta_i*sigma_i)

        #Bookkeeping
        self.ui_list.append(ui)
        self.user_to_selected[ui].append(x_i)

        return x_i

    def receive_reward(self, yif, yig):
        i = len(self.yi_list)
        ui = self.ui_list[-1]
        item = self.user_to_selected[ui][-1]
        oi = len(self.user_to_selected[ui]) - 1
        T_ui = self.Tui_dict[ui]

        mod = self.mod_func(item)

        distortion = (1 - 1.0/T_ui)**(T_ui - (oi+1))
        yi = distortion* yif + yig + (1 - distortion)*mod
        self.yi_list.append(yi)
        G, KGG_inv, Lambda, y_tilde = self.kernel_updates(i)
        return True

    def instantiate_kernel_matrices(self):
        self.G = np.array([0])
        self.proba_sampling_anchors = np.ones(self.G.shape[0])
        K_ZZ = np.array([[self.k1 + self.k2 + self.k3]])
        K_ZZ += EPS * np.eye(K_ZZ.shape[0])
        self.KGG_inv = np.linalg.inv(K_ZZ)
        self.KGG = K_ZZ
        self.K_ZS = np.array([[self.k1 + self.k2 + self.k3]])
        K_Z_s = self.K_ZS

        reward = self.yi_list[-1]
        self.Lambda = np.linalg.inv(np.dot(K_Z_s, K_Z_s.T) + self.reg_lambda * K_ZZ)
        self.y_tilde= np.array(reward) * K_Z_s

        self.proba_sampling_anchors = np.ones(self.G.shape[0])
        mu = self.mu_kors
        self.inv_kors_matrix = np.linalg.inv(K_ZZ + mu * np.eye(K_ZZ.shape[0]))
        return

    def kernel_updates(self, i):
        if i == 0:
            self.instantiate_kernel_matrices()
            return self.G, self.KGG_inv, self.Lambda, self.y_tilde

        G_prime, is_dict_updated = self.get_updated_dictionary(i)
        y_i = self.yi_list[-1]

        ui = self.ui_list[-1]
        x = self.user_to_selected[ui][-1]
        K_Ss = self.kernel_evaluate(range(i + 1), ui, x)

        #Update K_ZS with the new state by adding a column. This happens whether or not the dictionary is updated.
        self.K_ZS = np.hstack((self.K_ZS, self.K_Zs.reshape(self.K_ZS.shape[0], -1)))

        if not is_dict_updated:
            #Incremental inverse update for lambda_i with s_i
            self.Lambda = sherman_morrison_update(self.Lambda, self.K_Zs, self.K_Zs)
            self.y_tilde = self.y_tilde + y_i * self.K_Zs

        else:
            #Update K_Zs with the new anchor point by adding a row
            self.K_ZS = np.vstack((self.K_ZS, K_Ss))

            #Incremental inverse update for lambda_i with s_i, s
            #self.Lambda = np.linalg.inv(np.dot(self.K_ZS, self.K_ZS.T) + self.reg_lambda * np.eye(self.K_ZS.shape[0]))
            Lambda_Z = sherman_morrison_update(self.Lambda, self.K_Zs, self.K_Zs)
            a = self.K_ZS.dot(K_Ss).flatten()
            c = self.reg_lambda * self.kss + a[-1]
            b = self.reg_lambda * self.K_Zs + a[0:-1]
            self.Lambda = schur_first_order_update(Lambda_Z, b, c)

            #Note that, depending on the choice of kernel, if the eigenvalues of the matrices get really small,
            # then, self.Lambda may diverge from expected. But if a regularizer is added to the kernel,
            # then this should not be a problem.
            #expected = np.linalg.inv(np.dot(self.K_ZS, self.K_ZS.T) + self.reg_lambda * self.KGG)

            #Incremental inverse update for KGG_inv with s
            self.KGG_inv = schur_first_order_update(self.KGG_inv, self.K_Zs, self.kss)

            #Update expression for y_tilde
            self.y_tilde = self.y_tilde + y_i * self.K_Zs

            self.y_tilde = np.append(self.y_tilde, np.dot(self.yi_list, K_Ss))



        return G_prime, self.KGG_inv, self.Lambda, self.y_tilde

    def get_updated_dictionary(self, i):
        """
        This function updates the Nystrom dictionary
        of anchor points using the Kernel Online Row Sampling (KORS)
        technique.
        """
        mu, eps = self.mu_kors, self.epsilon_kors
        gamma = self.beta_lst[i] * self.reg_lambda
        A = self.inv_kors_matrix
        #sqprob = np.expand_dims(np.sqrt(self.proba_sampling_anchors), axis = 1)
        sqprob = np.sqrt(self.proba_sampling_anchors)


        ui = self.ui_list[-1]
        x = self.user_to_selected[ui][-1]
        self.K_Zs = self.kernel_evaluate(self.G, ui, x)
        b = self.K_Zs / sqprob

        self.kss = self.k1 + self.k2 + self.k3
        z = A.dot(b)
        u = b.T.dot(z)
        s = 1 / (self.kss + mu - u)
        tau = ((1+eps)/mu) * ( u + s * ( u - self.kss) * (u - self.kss))
        print("Tau:" + str(tau))
        p = max(min(gamma * tau, 1), 0)
        p = np.array(p).reshape(1)
        update = np.random.binomial(1, p)
        if update:
            self.proba_sampling_anchors = np.concatenate([self.proba_sampling_anchors, p])
            self.inv_kors_matrix = schur_first_order_update_fast(self.inv_kors_matrix, z /np.sqrt(p), b/np.sqrt(p), self.kss/p + mu)
            print("Updating with index:" + str(i))
            self.G = np.append(self.G, i)
            self.KGG = np.vstack((self.KGG, self.K_Zs))
            new_vec = np.append(self.K_Zs, self.kss).reshape(-1, 1)
            self.KGG = np.hstack((self.KGG, new_vec))
            return self.G, True
        else:
            return self.G, False

    def kernel_evaluate(self, G, ui, x):
        """
        This function computes the vector of kernel similarities
        for the current (context, action) pair with the collection
        of all historical (context,action) pairs
        for the timesteps contained in the set G.
        """
        movie_sim, meta_user_centers, k1, k2, k3, k4 = self.movie_sim, self.meta_user_centers, self.k1, self.k2, self.k3, self.k4
        ui_lst, Si_dict = self.ui_list, self.user_to_selected

        i = len(ui_lst) - 1
        history_one = Si_dict[ui][:-1]
        oi = len(Si_dict[ui]) - 1
        retlist = []

        for j in G:
            uj = ui_lst[j]
            oj = np.sum(ui_lst[:j] == uj)
            history_two = Si_dict[uj][:oj]
            movie_two = Si_dict[uj][oj]

            new_sim = composite_kernel_with_user_sep(x, movie_two, history_one, history_two, ui, uj, movie_sim, meta_user_centers, k1, k2, k3, k4, oi, oj)
            retlist.append(new_sim)

        return np.array(retlist)
