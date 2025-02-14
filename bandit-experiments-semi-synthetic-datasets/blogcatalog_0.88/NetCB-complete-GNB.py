from operator import itemgetter
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
import os
import time
import torch
import numpy as np
from numpy import *
from copy import deepcopy
import sklearn.metrics
from sklearn import datasets, linear_model, metrics
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as mp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import io
import random
import math
import _pickle as pickle
import torch
import time
from datetime import datetime
import sys
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from scipy.stats import linregress
from User_GNN_packages import *
import itertools


method = "NetCB-complete"
dte_rewards = []
against_flag = None
dte_activation_count = 0
variance_threshold = 0.00001
step_threshold = 300
against_threshold = 0.0
reg_slopes = []
#print(step_threshold, variance_threshold, against_threshold)

dataset = "blogcatalog" 

algo = "GNB"

activation_probability = [[0.5, 0.7]]
spillover_probability = [[0.3, 0.3]]

iteration_count = 5

hyper_param_1 = None
hyper_param_2 = None
hyper_param_3 =None
if dataset == "flickr":
  total_feature_count = 12047
  total_samples_count = 7575
  total_labels_count = 9
  if method == "CMAB":
    if algo == "LinUCB":
      hyper_param_1 = 0.4
    elif algo == "NeuralUCB":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.01
    elif algo == "NeuralTS":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.001
    elif algo == "EENet":
      hyper_param_1 = 0.001
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.1
      hyper_param_2 = 5
  else:
    if algo == "LinUCB":
      hyper_param_1 = 0.7
    elif algo == "NeuralUCB":
      hyper_param_1 = 0.1
      hyper_param_2 = 0.1
    elif algo == "NeuralTS":
      hyper_param_1 = 1
      hyper_param_2 = 0.001
    elif algo == "EENet":
      hyper_param_1 = 0.001
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.01
      hyper_param_2 = 5
elif dataset == "pubmed":
  total_feature_count = 500
  total_samples_count = 19717
  total_labels_count = 3
  if method == "CMAB":
    if algo == "LinUCB":
      hyper_param_1 = 4
    elif algo == "NeuralUCB":
      hyper_param_1 = 1
      hyper_param_2 = 0.001
    elif algo == "NeuralTS":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.01
    elif algo == "EENet":
      hyper_param_1 = 0.001
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.01
      hyper_param_2 = 5
  else:
    if algo == "LinUCB":
      hyper_param_1 = 2
    elif algo == "NeuralUCB":
      hyper_param_1 = 1
      hyper_param_2 = 0.01
    elif algo == "NeuralTS":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.001
    elif algo == "EENet":
      hyper_param_1 = 0.001
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.01
      hyper_param_2 = 5
elif dataset == "blogcatalog":
  total_feature_count = 8189
  total_samples_count = 5196
  total_labels_count = 6
  if method == "CMAB":
    if algo == "LinUCB":
      hyper_param_1 = 0.3
    elif algo == "NeuralUCB":
      hyper_param_1 = 0.1
      hyper_param_2 = 0.1
    elif algo == "NeuralTS":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.001
    elif algo == "EENet":
      hyper_param_1 = 0.001
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.01
      hyper_param_2 = 5
  else:
    if algo == "LinUCB":
      hyper_param_1 = 0.7
    elif algo == "NeuralUCB":
      hyper_param_1 = 0.1
      hyper_param_2 = 0.1
    elif algo == "NeuralTS":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.001
    elif algo == "EENet":
      hyper_param_1 = 0.001
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.01
      hyper_param_2 = 5
elif dataset == "hateful":
  total_feature_count = 1036
  total_samples_count = 3218
  total_labels_count = 2
  if method == "CMAB":
    if algo == "LinUCB":
      hyper_param_1 = 0.3
    elif algo == "NeuralUCB":
      hyper_param_1 = 1
      hyper_param_2 = 0.001
    elif algo == "NeuralTS":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.01
    elif algo == "EENet":
      hyper_param_1 = 0.1
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.1
      hyper_param_2 = 5
  else:
    if algo == "LinUCB":
      hyper_param_1 = 0.3
    elif algo == "NeuralUCB":
      hyper_param_1 = 1
      hyper_param_2 = 0.01
    elif algo == "NeuralTS":
      hyper_param_1 = 1
      hyper_param_2 = 0.001
    elif algo == "EENet":
      hyper_param_1 = 0.1
      hyper_param_2 = 0.001
      hyper_param_3 = 0.001
    elif algo == "GNB":
      hyper_param_1 = 0.01
      hyper_param_2 = 0.1

if dataset == "hateful":
  df_edges = np.load(open("edges", 'rb'), allow_pickle = True)
  df_edges = pd.DataFrame(df_edges, columns = ["target", "source"])
  df_nodes = np.load(open("features", 'rb'), allow_pickle = True)
  df_nodes = pd.DataFrame(df_nodes)
  y_LR = np.load(open("labels", 'rb'), allow_pickle = True)
elif dataset == "pubmed":
  features = pickle.load(open('attrs.pkl', 'rb'),  encoding='latin1')
  df_nodes = pd.DataFrame(features.toarray())
  df_edges = pd.read_csv('edgelist.txt', sep="\s+", header=None, names=["target", "source"])
  df_l = pd.read_csv('labels.txt', sep="\s+", header=None, names=["ID", "label"])
  y_LR = df_l.to_numpy()[:, 1]
  df_nodes.iloc[:, :] = np.load(open("df_nodes_pubmed_0.3", 'rb'), allow_pickle = True)
  y_LR = np.load(open("y_LR_pubmed_0.3", 'rb'), allow_pickle = True)
else:
  features = pickle.load(open('attrs.pkl', 'rb'),  encoding='latin1')
  df_nodes = pd.DataFrame(features.toarray())
  #df_edges = pd.read_csv('edgelist.txt', sep="\s+", header=None, names=["target", "source"])
  df_l = pd.read_csv('labels.txt', sep="\s+", header=None, names=["ID", "label"])
  y_LR = df_l.to_numpy()[:, 1]
  df_edges = np.load(open("df_edges_0.88", 'rb'), allow_pickle = True)
  df_edges = pd.DataFrame(df_edges, columns = ["target", "source"])
  
y_LR = [x - 1 for x in y_LR]

if total_feature_count > 500:
  X = df_nodes.to_numpy()
  svd = TruncatedSVD(n_components=500, n_iter=7, random_state=42)
  reduced_featurematrix = svd.fit_transform(X)
  df_nodes = pd.DataFrame(reduced_featurematrix)
  total_feature_count = 500

G = nx.Graph()
G = nx.from_pandas_edgelist(df_edges, source='source', target='target')

n_arms = total_labels_count
df_nodes_copy = df_nodes.copy(deep=True)
total_feature_count = total_feature_count + 4*n_arms

for arm_no in range(0, n_arms):
  df_nodes.insert(total_feature_count - (4*n_arms - arm_no), str(arm_no) + "_treated", [0.0]*total_samples_count, True)
  df_nodes_copy.insert(total_feature_count - (4*n_arms - arm_no), str(arm_no) + "_treated", [0]*total_samples_count, True)
for arm_no in range(0, n_arms):
  df_nodes.insert(total_feature_count - (3*n_arms - arm_no), str(arm_no) + "_treated_not_activated_nodes", [0.0]*total_samples_count, True)
  df_nodes_copy.insert(total_feature_count - (3*n_arms - arm_no), str(arm_no) + "_treated_not_activated_nodes", [0]*total_samples_count, True)

for arm_no in range(0, n_arms):
  df_nodes.insert(total_feature_count - (2*n_arms - arm_no), str(arm_no) + "_treated", [0.0]*total_samples_count, True)
  df_nodes_copy.insert(total_feature_count - (2*n_arms - arm_no), str(arm_no) + "_treated", [0]*total_samples_count, True)

for arm_no in range(0, n_arms):
  df_nodes.insert(total_feature_count - (n_arms - arm_no), str(arm_no) + "_treated_not_activated_nodes", [0.0]*total_samples_count, True)
  df_nodes_copy.insert(total_feature_count - (n_arms - arm_no), str(arm_no) + "_treated_not_activated_nodes", [0]*total_samples_count, True)

probability_treatment_node = None
probability_control_node = None


node_IDs = np.load(open("node_IDs", 'rb'), allow_pickle = True)

similar_arm_context = 0
opposite_arm_context = 0

changing_arm = None


treatment_rewards = 0
spillover_rewards = 0

d = total_feature_count
node_outcomes = [-1] * total_samples_count
flag_id = [-1] * total_samples_count         
treatment_id = [-1] * total_samples_count  
arm_type = [-1] * total_samples_count        
predicted_arm = [-1] * total_samples_count   

highest_spillover = None
lowest_spillover = None


class User_GNN_Bandit_Per_Arm:
    def __init__(self, low, high, low_activation, high_activation, dim, user_n, arm_n, k=1, GNN_lr=0.0001, user_lr=0.0001, hidden=100, bw_reward=10, bw_conf_b=10,
                 user_side=0, batch_size=-1, GNN_pooling_step_size=500, user_pooling_step_size=500,
                 arti_explore_constant=0.01, num_layer=-1, explore_param=1,
                 neighborhood_size=-1, train_every_user_model=False, separate_explore_GNN=False,
                 last_layer_gradient_flag=False,
                 device=None):
        global node_outcomes
        global flag_id
        global treatment_id
        global arm_type
        global highest_spillover
        global lowest_spillover
        global probability_treatment_node
        global probability_control_node
        global predicted_arm

        node_outcomes = [-1] * total_samples_count
        flag_id = [-1] * total_samples_count          
        treatment_id = [-1] * total_samples_count  
        arm_type = [-1] * total_samples_count        
        predicted_arm = [-1] * total_samples_count        
        highest_spillover = high
        lowest_spillover = low
        probability_treatment_node = high_activation
        probability_control_node = low_activation


        self.context_list = []
        self.reward = []
        self.GNN_lr = GNN_lr
        self.dim = dim
        self.hidden = hidden
        self.t = 0
        self.k = k
        self.batch_size = batch_size
        self.GNN_pooling_step_size = GNN_pooling_step_size
        self.arti_explore_constant = arti_explore_constant
        self.num_layer = num_layer
        self.model_explore_hidden = 100
        self.explore_param = explore_param
        self.neighborhood_size = neighborhood_size
        self.gpy_rbf_kernel_est = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.gpy_rbf_kernel_CB = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.device = device

        self.user_side = user_side

        self.bw_reward = bw_reward
        self.bw_conf_b = bw_conf_b

        self.user_n = user_n
        #
        if neighborhood_size > 0:
            graph_user_n = neighborhood_size
        else:
            graph_user_n = user_n

        self.graph_user_n = graph_user_n

        self.user_select_count = [0 for _ in range(user_n)]
        self.selected_user_period = set()
        self.arm_n = arm_n
        self.u_funcs_f_1 = {}
        self.u_funcs_f_2 = {}
        self.user_ests = None
        self.user_gradients = None
        self.separate_explore_GNN = separate_explore_GNN
        self.train_every_user_model = train_every_user_model
        self.last_layer_gradient_flag = last_layer_gradient_flag
        self.target_user_new_indices_ests, self.target_user_new_indices_CBs = None, None
        self.user_neighborhood_list_est = [[] for _ in range(arm_n)]
        self.selected_user_neighborhood_list = []

        # Dimension reduction operators
        self.GNN_reduced_grad_dim = user_n - 1
        self.user_reduced_grad_dim = arm_n - 1
        self.GNN_grad_op = LocallyLinearEmbedding(n_components=self.GNN_reduced_grad_dim)
        self.user_grad_op = LocallyLinearEmbedding(n_components=self.user_reduced_grad_dim)

        # Two user graphs
        self.user_exploitation_graph_dict = {i: np.zeros([graph_user_n, graph_user_n]) for i in range(arm_n)}
        self.user_exploration_graph_dict = {i: np.zeros([graph_user_n, graph_user_n]) for i in range(arm_n)}
        self.arm_to_target_user_dict_est = {}
        self.arm_to_target_user_dict_CB = {}
        #
        for a_i in range(arm_n):
            for i in range(graph_user_n):
                for j in range(i, graph_user_n):
                    weight_1 = 1 if i == j else random.random()
                    weight_2 = 1 if i == j else random.random()
                    self.user_exploitation_graph_dict[a_i][i, j] = weight_1
                    self.user_exploration_graph_dict[a_i][i, j] = weight_2
        #
        self.adj_m_exploit = []
        self.adj_m_explore = []
        #
        self.embedded_c_matrix = {}
        self.context_tensors = {}

        # Change the input dim with dimension reduction
        self.GNN_exploit_model = Exploitation_GNN(user_n=user_n, input_dim=self.dim,
                                                  reduced_output_dim=self.GNN_reduced_grad_dim, hidden_size=self.hidden,
                                                  lr_rate=GNN_lr, batch_size=batch_size,
                                                  pool_step_size=GNN_pooling_step_size, num_layer=num_layer,
                                                  last_layer_gradient_flag=last_layer_gradient_flag,
                                                  neighborhood_size=neighborhood_size,
                                                  device=device)
        if last_layer_gradient_flag:
            self.GNN_exploit_model.exploitation_model.change_grad_last_layer(predicting=True)

            GNN_total_param_count = sum(param.numel() for param in
                                        self.GNN_exploit_model.exploitation_model.est_module.parameters())
            self.GNN_exploit_model.exploitation_model.change_grad_last_layer(predicting=False)
        else:
            GNN_total_param_count = sum(param.numel() for param in
                                        self.GNN_exploit_model.exploitation_model.parameters())
        if self.GNN_pooling_step_size > 0:
            self.GNN_reduced_grad_dim = (GNN_total_param_count // self.GNN_pooling_step_size) + 1

        #
        self.GNN_explore_model = Exploration_GNN(user_n=user_n, input_dim=self.GNN_reduced_grad_dim,
                                                 hidden_size=self.model_explore_hidden,
                                                 lr_rate=GNN_lr, batch_size=batch_size,
                                                 separate_explore_GNN=self.separate_explore_GNN,
                                                 num_layer=num_layer,
                                                 neighborhood_size=neighborhood_size,
                                                 device=device)

        # ----------------------------------------------------------
        user_total_param_count = utils.getuser_f_1_param_count(dim, user_n, arm_n, self.user_reduced_grad_dim, hidden,
                                                               user_lr, batch_size, 1, device)

        #
        user_explore_grad_dim = int((user_total_param_count // user_pooling_step_size))
        if user_pooling_step_size > 0:
            self.user_reduced_grad_dim = user_explore_grad_dim



        for i in range(user_n):
            self.u_funcs_f_1[i] = Exploitation_FC(dim, user_n, arm_n=arm_n, reduced_dim=self.user_reduced_grad_dim,
                                                  hidden=hidden, lr=user_lr, batch_size=batch_size,
                                                  pool_step_size=user_pooling_step_size, device=device)
            #
            self.u_funcs_f_2[i] = Exploration_FC(self.user_reduced_grad_dim, hidden=self.model_explore_hidden,
                                                 lr=user_lr, batch_size=batch_size, device=device)

        self.exploitation_adj_matrix_dict, self.exploration_adj_matrix_dict = None, None
        self.selected_arm = None


    def update_info(self, u_selected, a_selected, contexts, reward, GNN_gradient, GNN_residual_reward):
        #
        self.user_select_count[u_selected] += 1
        self.selected_user_period.add(u_selected)

        # Update EE-Net module info
        reward = torch.tensor(reward)
        context = torch.tensor(contexts[a_selected, :])

        user_gradient = self.user_gradients[u_selected][a_selected, :].detach().reshape(-1, )
        user_residual_reward = reward - self.user_ests[u_selected, a_selected].detach()

        self.u_funcs_f_1[u_selected].update(context, reward)
        self.u_funcs_f_2[u_selected].update(user_gradient, user_residual_reward)

        # Update GNN module info
        embed_c = self.embedded_c_matrix[a_selected]

        if self.separate_explore_GNN:
            embed_g = torch.tensor(
                utils.generate_matrix_embedding_gradients(source=GNN_gradient)).float()
        else:
            embed_g = GNN_gradient

        GNN_residual_reward = GNN_residual_reward
        exploit_adj_m_tensor = self.exploitation_adj_matrix_dict[a_selected]
        explore_adj_m_tensor = self.exploration_adj_matrix_dict[a_selected]

        #
        if self.neighborhood_size > 0:
            u_selected_tensor = self.target_user_new_indices_ests[a_selected]
        else:
            u_selected_tensor = torch.tensor(np.array([u_selected]))

        self.GNN_exploit_model.update_info(embed_c, reward, u_selected_tensor, exploit_adj_m_tensor,
                                           selected_neighborhood=self.user_neighborhood_list_est[a_selected])
        self.GNN_explore_model.update_info(embed_g, GNN_residual_reward, u_selected_tensor, explore_adj_m_tensor)

    def update_artificial_explore_info(self, t, u_selected, arm_selected, whole_gradients):
        index = 0
        # u_selected_tensor = torch.tensor(np.array([u_selected]))
        '''set small scores for un-selected arms if the selected arm is 0-reward'''
        # c = torch.tensor([1 / np.log(1 * t + 10000)]).float() --- MNIST-only
        # c = torch.tensor([1 / np.log(1 * t + 10)]).float()
        c = torch.tensor(np.array([self.arti_explore_constant]))
        for arm_grad in whole_gradients:
            if index != arm_selected:
                explore_adj_m_tensor = self.exploration_adj_matrix_dict[index]
                if self.neighborhood_size > 0:
                    u_selected_tensor = self.target_user_new_indices_ests[index]
                else:
                    u_selected_tensor = torch.tensor(np.array([u_selected]))

                #
                if self.separate_explore_GNN:
                    embed_g = torch.tensor(
                        utils.generate_matrix_embedding_gradients(source=arm_grad)).float()
                else:
                    # embed_g = torch.tensor(arm_grad).float()
                    embed_g = arm_grad

                user_gradient = self.user_gradients[u_selected][index, :].detach().reshape(-1, )

                #
                self.GNN_explore_model.update_info(embed_g, c, u_selected_tensor, explore_adj_m_tensor)
                self.u_funcs_f_2[u_selected].update(user_gradient, c)

            index += 1

    ############################################################################
    def get_top_users_random(self, reward_ests, CB_ests, target_user):
        top_k_est_tensor, top_k_CB_tensor = [], []
        target_user_new_indices_ests, target_user_new_indices_CBs = [], []
        user_neighborhood_list_est = []
        target_user_tensor = torch.ones(1, ) * target_user
        user_range = [*range(target_user), *range(target_user + 1, self.user_n)]

        for a_i in range(self.arm_n):
            #
            # sampled_users = torch.tensor(np.random.choice(user_range, size=self.neighborhood_size-1))
            sampled_users = torch.arange(start=0, end=self.neighborhood_size).long()
            sampled_users = torch.cat([target_user_tensor, sampled_users]).long()
            top_users_combined = torch.unique(sampled_users, sorted=True).reshape(-1, ).to(self.device)
            new_index = (top_users_combined == target_user).nonzero(as_tuple=False).reshape(-1, ).to(self.device)

            # indices combined
            top_k_est_tensor.append(reward_ests[top_users_combined, a_i].reshape(-1, 1))
            top_k_CB_tensor.append(CB_ests[top_users_combined, a_i].reshape(-1, 1))
            user_neighborhood_list_est.append(top_users_combined[:, None].to(self.device))

            #
            target_user_new_indices_ests.append(new_index)
            target_user_new_indices_CBs.append(new_index)

        return top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
               user_neighborhood_list_est

    ############################################################################
    def get_top_users_most_frequent(self, reward_ests, CB_ests, target_user):
        top_k_est_tensor, top_k_CB_tensor = [], []
        target_user_new_indices_ests, target_user_new_indices_CBs = [], []
        user_neighborhood_list_est = []
        target_user_tensor = torch.ones(1, ) * target_user

        user_range = np.array([*range(target_user), *range(target_user + 1, self.user_n)])
        user_count = torch.tensor(np.array(self.user_select_count)[user_range])
        (_, top_user_est_i) = torch.topk(user_count, k=self.neighborhood_size-1, largest=True)
        sampled_users = top_user_est_i

        new_index = torch.zeros(1).long().to(self.device)
        sampled_users = torch.cat([target_user_tensor, sampled_users]).long()
        for a_i in range(self.arm_n):
            # indices combined
            top_users_combined = sampled_users

            #
            top_k_est_tensor.append(reward_ests[top_users_combined, a_i].reshape(-1, 1))
            top_k_CB_tensor.append(CB_ests[top_users_combined, a_i].reshape(-1, 1))
            user_neighborhood_list_est.append(top_users_combined[:, None].to(self.device))

            #
            target_user_new_indices_ests.append(new_index)
            target_user_new_indices_CBs.append(new_index)

        return top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
               user_neighborhood_list_est

    # ############################################################################
    def get_top_users(self, reward_ests, CB_ests, target_user=-1):
        top_k_est_tensor, top_k_CB_tensor = [], []
        target_user_new_indices_ests, target_user_new_indices_CBs = [], []
        user_neighborhood_list_est = []
        for a_i in range(self.arm_n):
            #
            other_user_ests, other_user_CBs = reward_ests[:, a_i], CB_ests[:, a_i]
            diff_ests, diff_CBs \
                = torch.abs(other_user_ests - reward_ests[target_user, a_i]).reshape(-1, ), \
                  torch.abs(other_user_CBs - CB_ests[target_user, a_i]).reshape(-1, )
            (_, top_user_est_i) = torch.topk(diff_ests, k=self.neighborhood_size, largest=False)
            (_, top_user_CB_i) = torch.topk(diff_CBs, k=self.neighborhood_size, largest=False)

            # indices combined
            top_users_combined = torch.cat([top_user_est_i, top_user_CB_i])
            top_users_combined = torch.unique(top_users_combined, sorted=True).reshape(-1, )

            #
            top_k_est_tensor.append(reward_ests[top_users_combined, a_i].reshape(-1, 1))
            top_k_CB_tensor.append(CB_ests[top_users_combined, a_i].reshape(-1, 1))
            user_neighborhood_list_est.append(top_users_combined[:, None].to(self.device))

            #
            new_index = (top_users_combined == target_user).nonzero(as_tuple=False).reshape(-1, )
            target_user_new_indices_ests.append(new_index)
            target_user_new_indices_CBs.append(new_index)

        return top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
               user_neighborhood_list_est

    def update_user_graphs(self, contexts, user_i, random_user_flag=False):
        reward_ests = []
        CB_ests = []
        gradients = []
        n_arms = contexts.shape[0]

        #
        top_k_est_tensor, top_k_CB_tensor = None, None
        if self.neighborhood_size > 0:
            #
            for u_i in range(self.user_n):
                res, grad = self.u_funcs_f_1[u_i].output_and_gradient(context=contexts)
                exp_scores = self.u_funcs_f_2[u_i].output(context=grad)
                reward_ests.append(res.reshape(-1, ))
                CB_ests.append(exp_scores.reshape(-1, ))
                gradients.append(grad.reshape(n_arms, -1))
            #
            reward_ests = torch.stack(reward_ests, dim=0)
            CB_ests = torch.stack(CB_ests, dim=0)
            #
            top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
                self.user_neighborhood_list_est = self.get_top_users_random(reward_ests, CB_ests, target_user=user_i)
            self.target_user_new_indices_ests, self.target_user_new_indices_CBs = \
                target_user_new_indices_ests, target_user_new_indices_CBs
            self.user_ests = reward_ests
            self.user_gradients = gradients
        else:
            #
            for u_i in range(self.user_n):
                res, grad = self.u_funcs_f_1[u_i].output_and_gradient(context=contexts)
                exp_scores = self.u_funcs_f_2[u_i].output(context=grad)
                reward_ests.append(res.reshape(-1, ))
                CB_ests.append(exp_scores.reshape(-1, ))
                gradients.append(grad.reshape(n_arms, -1))
            #
            reward_ests = torch.stack(reward_ests, dim=0)
            CB_ests = torch.stack(CB_ests, dim=0)
            gradients = torch.stack(gradients, dim=0)
            self.user_ests = reward_ests
            self.user_gradients = gradients
            reward_ests = reward_ests.detach().cpu().numpy()
            CB_ests = CB_ests.detach().cpu().numpy()

        # Update two graphs
        for a_i in range(self.arm_n):
            if self.neighborhood_size > 0:
                this_reward_ests, this_CB_ests = \
                    top_k_est_tensor[a_i].detach().cpu().numpy(), top_k_CB_tensor[a_i].detach().cpu().numpy()
                #
                self.user_exploitation_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(this_reward_ests, this_reward_ests, self.bw_reward)).to(self.device)
                self.user_exploration_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(this_CB_ests, this_CB_ests, self.bw_conf_b)).to(self.device)
            else:
                self.user_exploitation_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(reward_ests, reward_ests, self.bw_reward)).to(self.device)
                self.user_exploration_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(CB_ests, CB_ests, self.bw_conf_b)).to(self.device)

    def get_normalized_adj_m_list_for_user_graphs(self):
        exploitation_adj_matrix_dict = {}
        exploration_adj_matrix_dict = {}

        #
        if self.neighborhood_size > 0:
            for a_i in range(self.arm_n):
                exploitation_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploitation_graph_dict[a_i], k=self.k)
                exploitation_adj_matrix_dict[a_i] = exploitation_adj_matrix_normalized
                #
                exploration_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploration_graph_dict[a_i], k=self.k)
                exploration_adj_matrix_dict[a_i] = exploration_adj_matrix_normalized
        else:
            for a_i in range(self.arm_n):
                exploitation_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploitation_graph_dict[a_i], k=self.k)
                exploitation_adj_matrix_dict[a_i] = exploitation_adj_matrix_normalized
                #
                exploration_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploration_graph_dict[a_i], k=self.k)
                exploration_adj_matrix_dict[a_i] = exploration_adj_matrix_normalized

        return exploitation_adj_matrix_dict, exploration_adj_matrix_dict

    def train_user_models(self, u):
        if self.train_every_user_model:
            exploit_loss, explore_loss = 0, 0
            if self.batch_size <= 0:
                for u_i in self.selected_user_period:
                    exploit_loss = self.u_funcs_f_1[u_i].train()
                    explore_loss = self.u_funcs_f_2[u_i].train()
            else:
                for u_i in self.selected_user_period:
                    exploit_loss = self.u_funcs_f_1[u_i].batch_train()
                    explore_loss = self.u_funcs_f_2[u_i].batch_train()
            self.selected_user_period = set()
        # -------------------------------------------------
        else:
            if self.batch_size <= 0:
                exploit_loss = self.u_funcs_f_1[u].train()
                explore_loss = self.u_funcs_f_2[u].train()
            else:
                exploit_loss = self.u_funcs_f_1[u].batch_train()
                explore_loss = self.u_funcs_f_2[u].batch_train()

        return exploit_loss, explore_loss

    def train_GNN_models(self):
        exploit_adj_tensor = self.exploitation_adj_matrix_dict[self.selected_arm]
        explore_adj_tensor = self.exploration_adj_matrix_dict[self.selected_arm]

        if self.batch_size <= 0:
            exploit_loss = self.GNN_exploit_model.train_model(c_adj_m=exploit_adj_tensor)
            explore_loss = self.GNN_explore_model.train_model(c_adj_m=explore_adj_tensor)
        else:
            exploit_loss = self.GNN_exploit_model.train_model_batch(c_adj_m=exploit_adj_tensor)
            explore_loss = self.GNN_explore_model.train_model_batch(c_adj_m=explore_adj_tensor)

        return exploit_loss, explore_loss

    def recommend(self, u, contexts, t):
        self.t = t
        g_list = []
        res_list = []
        overall_ests_list = []
        u_tensor = torch.tensor(np.array([u])).to(self.device)

        # Get adjacency matrices for user graphs
        self.exploitation_adj_matrix_dict, self.exploration_adj_matrix_dict = \
            self.get_normalized_adj_m_list_for_user_graphs()

        # Reward estimation ---------------------------------------------
        reduced_grad_array = []
        for a_i, c in enumerate(contexts):

            exploit_adj_m_tensor = self.exploitation_adj_matrix_dict[a_i]
            this_user_n = exploit_adj_m_tensor.shape[0]
            tensor = utils.generate_matrix_embedding_user(source=c, user_n=this_user_n).to(self.device)
            self.embedded_c_matrix[a_i] = tensor
            self.context_tensors[a_i] = torch.tensor(c).to(self.device)

            # f_1
            users_res, users_g \
                = self.GNN_exploit_model.get_reward_estimate_and_gradients(contexts=tensor, adj_m=exploit_adj_m_tensor,
                                                                           neighborhood_users=self.user_neighborhood_list_est[a_i])
            #
            if self.neighborhood_size > 0:
                user_i = self.target_user_new_indices_ests[a_i]
            else:
                user_i = u_tensor
            r_est = users_res[user_i]
            res_list.append(r_est)

            #
            users_g = F.avg_pool1d(users_g.unsqueeze(0), kernel_size=self.GNN_pooling_step_size,
                                   stride=self.GNN_pooling_step_size, ceil_mode=True).squeeze(0)
            #
            reduced_grad_array.append(users_g)

        #
        for a_i in range(self.arm_n):
            explore_adj_m_tensor = self.exploration_adj_matrix_dict[a_i]

            #
            users_g = reduced_grad_array[a_i]
            gradients_tensor = users_g

            #
            if self.neighborhood_size > 0:
                user_i = self.target_user_new_indices_CBs[a_i]
            else:
                user_i = u_tensor
            explore_s = self.GNN_explore_model.get_exploration_scores(gradients=gradients_tensor,
                                                                      adj_m=explore_adj_m_tensor, user_i=user_i,
                                                                      user_neighborhood=self.user_neighborhood_list_est[a_i])

            # f_1 + f_2
            r_est = res_list[a_i]
            sample_r = r_est + (self.explore_param * explore_s)
            overall_ests_list.append(sample_r.item())
            g_list.append(users_g)
        #
        selected_arm = np.argmax(overall_ests_list)
        point_est = res_list[selected_arm]

        self.selected_arm = selected_arm
        self.exploit_adj_m_normalized = self.exploitation_adj_matrix_dict[selected_arm]
        self.explore_adj_m_normalized = self.exploration_adj_matrix_dict[selected_arm]

        return selected_arm, g_list[selected_arm], point_est, g_list
        
    def recommend_without_update(self, u, contexts):
        g_list = []
        res_list = []
        overall_ests_list = []
        u_tensor = torch.tensor(np.array([u])).to(self.device)

        # Get adjacency matrices for user graphs
        self.exploitation_adj_matrix_dict, self.exploration_adj_matrix_dict = \
            self.get_normalized_adj_m_list_for_user_graphs()

        # Reward estimation ---------------------------------------------
        reduced_grad_array = []
        for a_i, c in enumerate(contexts):

            exploit_adj_m_tensor = self.exploitation_adj_matrix_dict[a_i]
            this_user_n = exploit_adj_m_tensor.shape[0]
            tensor = utils.generate_matrix_embedding_user(source=c, user_n=this_user_n).to(self.device)
            self.embedded_c_matrix[a_i] = tensor
            self.context_tensors[a_i] = torch.tensor(c).to(self.device)

            # f_1
            users_res, users_g \
                = self.GNN_exploit_model.get_reward_estimate_and_gradients(contexts=tensor, adj_m=exploit_adj_m_tensor,
                                                                           neighborhood_users=self.user_neighborhood_list_est[a_i])
            #
            if self.neighborhood_size > 0:
                user_i = self.target_user_new_indices_ests[a_i]
            else:
                user_i = u_tensor
            r_est = users_res[user_i]
            res_list.append(r_est)

            #
            users_g = F.avg_pool1d(users_g.unsqueeze(0), kernel_size=self.GNN_pooling_step_size,
                                   stride=self.GNN_pooling_step_size, ceil_mode=True).squeeze(0)
            #
            reduced_grad_array.append(users_g)

        #
        for a_i in range(self.arm_n):
            explore_adj_m_tensor = self.exploration_adj_matrix_dict[a_i]

            #
            users_g = reduced_grad_array[a_i]
            gradients_tensor = users_g

            #
            if self.neighborhood_size > 0:
                user_i = self.target_user_new_indices_CBs[a_i]
            else:
                user_i = u_tensor
            explore_s = self.GNN_explore_model.get_exploration_scores(gradients=gradients_tensor,
                                                                      adj_m=explore_adj_m_tensor, user_i=user_i,
                                                                      user_neighborhood=self.user_neighborhood_list_est[a_i])

            # f_1 + f_2
            r_est = res_list[a_i]
            sample_r = r_est + (self.explore_param * explore_s)
            overall_ests_list.append(sample_r.item())
            #g_list.append(users_g)
        #
        selected_arm = np.argmax(overall_ests_list)

        return selected_arm

def update_spillover_neighbours(id_source, id, is_treated = True): 
  node_neighbours = list(G.neighbors(id)) 
  if is_treated == True: 
    for j in node_neighbours:   
      for candidate_arm in range(0, n_arms):
        if flag_id[id_source] == candidate_arm:
          df_nodes_copy.iloc[j, total_feature_count - (4*n_arms - candidate_arm)] += 1  
          total = 0
          for arm_no in range(0, n_arms):
            total += df_nodes_copy.iloc[j, total_feature_count - (4*n_arms - arm_no)]
          for arm_no in range(0, n_arms):
            df_nodes.iloc[j, total_feature_count - (4*n_arms - arm_no)] = df_nodes_copy.iloc[j, total_feature_count - (4*n_arms - arm_no)] / total

        if flag_id[id_source] == candidate_arm and node_outcomes[id] == 0:
          df_nodes_copy.iloc[j, total_feature_count - (3*n_arms - candidate_arm)] += 1   
          total = 0
          for arm_no in range(0, n_arms):
            total += df_nodes_copy.iloc[j, total_feature_count - (3*n_arms - arm_no)]
          for arm_no in range(0, n_arms):
            df_nodes.iloc[j, total_feature_count - (3*n_arms - arm_no)] = df_nodes_copy.iloc[j, total_feature_count - (3*n_arms - arm_no)] / total


def update_neighbours(id, is_treated = True):
  node_neighbours = list(G.neighbors(id)) 
  if is_treated == True: 
    for j in node_neighbours:   
      for candidate_arm in range(0, n_arms):
        if flag_id[id] == candidate_arm:
          df_nodes_copy.iloc[j, total_feature_count - (2*n_arms - candidate_arm)] += 1  
          total = 0
          for arm_no in range(0, n_arms):
            total += df_nodes_copy.iloc[j, total_feature_count - (2*n_arms - arm_no)]
          for arm_no in range(0, n_arms):
            df_nodes.iloc[j, total_feature_count - (2*n_arms - arm_no)] = df_nodes_copy.iloc[j, total_feature_count - (2*n_arms - arm_no)] / total

        if flag_id[id] == candidate_arm and node_outcomes[id] == 0:
          df_nodes_copy.iloc[j, total_feature_count - (n_arms - candidate_arm)] += 1   
          total = 0
          for arm_no in range(0, n_arms):
            total += df_nodes_copy.iloc[j, total_feature_count - (n_arms - arm_no)]
          for arm_no in range(0, n_arms):
            df_nodes.iloc[j, total_feature_count - (n_arms - arm_no)] = df_nodes_copy.iloc[j, total_feature_count - (n_arms - arm_no)] / total

  elif is_treated == False:  
    for candidate_arm in range(0, n_arms):
      if flag_id[id] == candidate_arm:
        for j in node_neighbours: 
          df_nodes_copy.iloc[j, total_feature_count - (n_arms - candidate_arm)] -= 1  
          total = 0
          for arm_no in range(0, n_arms):
            total += df_nodes_copy.iloc[j, total_feature_count - (n_arms - arm_no)]
          for arm_no in range(0, n_arms):
            df_nodes.iloc[j, total_feature_count - (n_arms - arm_no)] = df_nodes_copy.iloc[j, total_feature_count - (n_arms - arm_no)] / total

def nullify(node_ID):
  for j in node_ID:
    for arm_no in range(0, n_arms):
      df_nodes_copy.iloc[j, total_feature_count - (2*n_arms - arm_no)] = 0
      df_nodes_copy.iloc[j, total_feature_count - (n_arms - arm_no)] = 0
      df_nodes.iloc[j, total_feature_count - (2*n_arms - arm_no)] = 0.0
      df_nodes.iloc[j, total_feature_count - (n_arms - arm_no)] = 0.0
      df_nodes_copy.iloc[j, total_feature_count - (4*n_arms - arm_no)] = 0
      df_nodes_copy.iloc[j, total_feature_count - (3*n_arms - arm_no)] = 0
      df_nodes.iloc[j, total_feature_count - (4*n_arms - arm_no)] = 0.0
      df_nodes.iloc[j, total_feature_count - (3*n_arms - arm_no)] = 0.0


def generate_outcome(arm, id, spillover_index, freq, model):
  global similar_arm_context
  global opposite_arm_context
  global Aa
  global Aa_inv
  global ba
  global theta
  global node_outcomes, flag_id, arm_type, predicted_arm, treatment_id
  global changing_arm
  global treatment_rewards
  global spillover_rewards
  global n_arms
  global dte_activation_count, dte_rewards, against_flag
  
  optimal_rewards = max_rewards(id) 
  prev_node_count = node_outcomes.count(1) 
  
  is_arm_changed = False

  context = y_LR[id]
  r = None
  flag = None
  prev_count = node_outcomes.count(1)
  
  if arm == context:
    similar_arm_context += 1
  else:
    opposite_arm_context += 1

  predicted_arm[id] = arm

  if against_flag == False and freq >= step_threshold:
    reg_slopes.append(linregress(np.arange(freq - step_threshold, freq), dte_rewards[freq - step_threshold : freq]).slope)
  if against_flag == False and len(reg_slopes) >= step_threshold:
    slope_flag = True
    for slp in range(len(reg_slopes)-step_threshold, len(reg_slopes)):
      if reg_slopes[slp] < -variance_threshold or reg_slopes[slp] > variance_threshold:
        slope_flag = False
        break
    if slope_flag == True:
      #print("against_start", freq)
      against_flag = True

  if against_flag == True:# and len(list(G.neighbors(id))) > 15: 
    #check whether going against the predicted arm is fruitful
    e_ij_p_high = highest_spillover
    e_ij_p_low = lowest_spillover
    DTE_right = probability_treatment_node
    DTE_wrong = probability_control_node

    expected_reward_arms = [0.0] * n_arms

    reward_predicted_arm = None
    reward_alternate_arm = None
  
    for label in range(n_arms):
      if label == arm:
        # m is the total number of inactive neighbors whose predicted arms are the same to that of predicted arm of the arrival node
        node_neighbours = list(G.neighbors(id)) 
        m = 0
        for j in node_neighbours:   
          if node_outcomes[j] == 0  and label == model.recommend_without_update(format_features(j, df_nodes.iloc[j], total_labels_count, total_feature_count)[0], format_features(j, df_nodes.iloc[j], total_labels_count, total_feature_count)[1]):  
            m += 1
        reward_predicted_arm = DTE_right + DTE_right * e_ij_p_high * m 
        expected_reward_arms[label] = reward_predicted_arm
      else:
        #where m_prime is the total number of inactive neighbors whose predicted arms are the opposite to that of predicted arm of the arrival node
        node_neighbours = list(G.neighbors(id)) 
        m_prime = 0
        for j in node_neighbours:  
          if node_outcomes[j] == 0  and label == model.recommend_without_update(format_features(j, df_nodes.iloc[j], total_labels_count, total_feature_count)[0], format_features(j, df_nodes.iloc[j], total_labels_count, total_feature_count)[1]):
            m_prime += 1
        reward_alternate_arm = DTE_wrong + DTE_wrong * e_ij_p_low * m_prime
        expected_reward_arms[label] = reward_alternate_arm

    highest_reward_label= np.argmax(expected_reward_arms)
    reward_alternate_arm = expected_reward_arms[highest_reward_label]

    test_diff[spillover_index].append(reward_predicted_arm - reward_alternate_arm)
    if (reward_alternate_arm - reward_predicted_arm > against_threshold):
      arm = highest_reward_label  
      # print("changing the arm") 
      changing_arm += 1
      is_arm_changed = True
      test_diff_alternating[spillover_index].append(reward_predicted_arm - reward_alternate_arm)

  if arm == context:
    r = np.array([1])
    flag = np.random.binomial(1, probability_treatment_node, 1)
    arm_type[id] = 1
  else:
    r = np.array([0])
    flag = np.random.binomial(1, probability_control_node, 1)
    arm_type[id] = 0

  node_outcomes[id] = flag[0]
  flag_id[id] = arm   
  treatment_id[id] = arm

  is_visited = [False] * total_samples_count
  is_visited[id] = True
  queue = []

  dte_activation_count += node_outcomes[id]
  prev_treatment_rewards = treatment_rewards
  update_neighbours(id, True)

  after_count = node_outcomes.count(1)
  #contagion to neighbour nodes
  if node_outcomes[id] == 1:
    node_neighbours = list(G.neighbors(id)) 
    for j in node_neighbours:   
      last_treatment_rewards = treatment_rewards
      if node_outcomes[j] != 1:
        treatment_rewards += 1
      if treatment_id[id] == y_LR[id] and node_outcomes[j] != 1:
        if (treatment_id[id] == y_LR[j]):
          node_outcomes[j] = np.random.binomial(1, highest_spillover)
          if node_outcomes[j] == 1:
            spillover_rewards += 1
            treatment_id[j] = treatment_id[id]  
            queue.append(j)
            is_visited[j] = True
      elif treatment_id[id] != y_LR[id] and node_outcomes[j] != 1:
        if (treatment_id[id] == y_LR[j]):
          node_outcomes[j] = np.random.binomial(1, lowest_spillover)
          if node_outcomes[j] == 1:
            spillover_rewards += 1
            treatment_id[j] = treatment_id[id]  
            queue.append(j)
            is_visited[j] = True 
      if last_treatment_rewards != treatment_rewards:
        update_spillover_neighbours(id, j, True) 
    after_count = node_outcomes.count(1)

  after_node_count = node_outcomes.count(1) 
  after_treatment_rewards = treatment_rewards
  predicted_rewards = after_node_count - prev_node_count 
  

  r[0] = after_count - prev_count
  
  if predicted_rewards > optimal_rewards:
    return r[0], is_arm_changed, 0
  else:
    return r[0], is_arm_changed, optimal_rewards - predicted_rewards 


def format_features(id, user_features, n_arm, act_dim):
  dim = act_dim + n_arm - 1
  X = np.zeros((n_arm, dim))
  target = y_LR[id]
  for a in range(n_arm):
    X[a, a:a + act_dim] = user_features
  rwd = np.zeros(n_arm)
  rwd[target] = 1
  return target, X, rwd 
  
def max_rewards(id): 
  context = y_LR[id]
  max_R = 0
  for label in range(n_arms):
    activated_count = 0
    if label == context:
      flag = np.random.binomial(1, probability_treatment_node, 1)
    else:
      flag = np.random.binomial(1, probability_control_node, 1)
    activated_count += flag[0]

    #contagion to neighbour nodes
    if flag[0] == 1:
      node_neighbours = list(G.neighbors(id)) 
      for j in node_neighbours:   
        node_outcomes_j = 0
        if label == y_LR[id] and node_outcomes[j] != 1:
          if (label == y_LR[j]):
            node_outcomes_j = np.random.binomial(1, highest_spillover)
        elif label != y_LR[id] and node_outcomes[j] != 1:
          if (label == y_LR[j]):
            node_outcomes_j = np.random.binomial(1, lowest_spillover)
        activated_count += node_outcomes_j
    if activated_count > max_R:
      max_R = activated_count
  return max_R
  
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


Changing_Arms_avg = [] #3D array
Reward_3D = []
Regret_3D = []
Period_3D = []
Reward_treatment_3D = []
Reward_spillover_3D = []
Right_arm_3D = []
regres_avg = []

for iter in range(iteration_count):
  dte_rewards = []
  reg_slopes = []
  dte_activation_count = 0
  against_flag = False
  node_ID = list(node_IDs[iter])

  RA_ratio = [] #2D array
  Reward = []  #2D array
  Regret = []  #2D array
  Reward_treatment = []  #2D array
  Reward_spillover = []  #2D array
  Right_arm = [] #2D array
  Period = [] # 2D array
  Changing_Arms = [] # 2D array

  l_spillover = None
  h_spillover = None
  incorrect_activation = None
  correct_activation = None

  for activation_index in range(len(activation_probability)):
    incorrect_activation = activation_probability[activation_index][0]
    correct_activation = activation_probability[activation_index][1]
    list_list_total_reward = []
    list_list_total_regret = []
    list_list_total_reward_treatment = []
    list_list_total_reward_spillover = []
    list_list_total_reward_action_ratio = []
    list_list_period = []
    list_list_right_arm = []
    list_list_changing_arm = [] # 1D array
    for spillover_index in range(len(spillover_probability)):
      l_spillover = spillover_probability[spillover_index][0]
      h_spillover = spillover_probability[spillover_index][1]
      nullify(node_ID)
      #print(l_spillover,h_spillover)
      similar_arm_context = 0
      opposite_arm_context = 0
      changing_arm = 0

      node_outcomes = [-1] * total_samples_count
      flag_id = [-1] * total_samples_count          
      treatment_id = [-1] * total_samples_count
      arm_type = [-1] * total_samples_count
      

      args_arti_explore_constant=0.1
      args_k = 1
      model = User_GNN_Bandit_Per_Arm(l_spillover , h_spillover, incorrect_activation, correct_activation, dim = total_feature_count + total_labels_count - 1, user_n=total_labels_count, arm_n=total_labels_count, k=1,
                                                GNN_lr=0.0001, user_lr=0.0001,
                                                bw_reward=hyper_param_2, bw_conf_b=hyper_param_2,
                                                batch_size=-1,
                                                GNN_pooling_step_size=1000,
                                                user_pooling_step_size=100,
                                                arti_explore_constant=hyper_param_1,
                                                num_layer=-1, explore_param=1,
                                                separate_explore_GNN=False,
                                                train_every_user_model=True,
                                                device=device)
                
      list_total_reward = []
      list_total_regret = []
      list_total_reward_treatment = []
      list_total_reward_spillover = []
      list_total_reward_action_ratio = []
      list_period = []
      list_right_arm = []
      period=0
      treatment_rewards = 0
      spillover_rewards = 0
      total_regrets = 0

      for i in tqdm(node_ID):
        df_nodes = df_nodes.fillna(0)
        user_features = df_nodes.iloc[i]
        if node_outcomes[i] != 1:
          u, contexts, rwd = format_features(i, user_features, total_labels_count, total_feature_count)
          model.update_user_graphs(contexts=contexts, user_i=u)

          arm_select, user_g, point_est, whole_gradients = model.recommend(u, contexts, period)

          r, is_arm_changed,regrets = generate_outcome(arm_select, i, spillover_index, period, model)
          total_regrets += regrets
          GNN_residual_reward = r - point_est

          if r == 0 and args_arti_explore_constant > 0 and is_arm_changed == False:
            model.update_artificial_explore_info(period, u, arm_select, whole_gradients)
          if is_arm_changed == False: 
            model.update_info(u_selected=u, a_selected=arm_select, contexts=contexts, reward=r, GNN_gradient=user_g, GNN_residual_reward=GNN_residual_reward)

          if period < 1000:
            if period % 10 == 0 and is_arm_changed == False:
              u_exploit_loss, u_explore_loss = model.train_user_models(u=u)
              GNN_exploit_loss, GNN_explore_loss = model.train_GNN_models()
          else:
            if period % 100 == 0 and is_arm_changed == False:
              u_exploit_loss, u_explore_loss = model.train_user_models(u=u)
              GNN_exploit_loss, GNN_explore_loss = model.train_GNN_models()

          if period % 10 == 0 and is_arm_changed == False:
            # Exploitation
            np_mat_exloit_no_norm = model.user_exploitation_graph_dict[arm_select].cpu().numpy()
            np_mat_exploit = model.exploit_adj_m_normalized.cpu().numpy()
            powered_mat = np.linalg.matrix_power(np_mat_exploit, max(1, args_k))
            off_diag_matrix = np.copy(powered_mat)
            np.fill_diagonal(off_diag_matrix, 0)
            off_diag_matrix_no_norm = np.copy(np_mat_exloit_no_norm)
            np.fill_diagonal(off_diag_matrix_no_norm, 0)

            # Exploration
            np_mat_explore_no_norm = model.user_exploration_graph_dict[arm_select].cpu().numpy()
            np_mat_explore = model.explore_adj_m_normalized.cpu().numpy()
            powered_mat = np.linalg.matrix_power(np_mat_explore, max(1, args_k))
            off_diag_matrix = np.copy(powered_mat)
            np.fill_diagonal(off_diag_matrix, 0)
            off_diag_matrix_no_norm = np.copy(np_mat_explore_no_norm)
            np.fill_diagonal(off_diag_matrix_no_norm, 0)

          period += 1
          
          dte_rewards.append(dte_activation_count/period)
          list_total_reward_treatment.append(dte_activation_count/period)
          if period % 10 == 0:
            list_total_reward.append(node_outcomes.count(1))
            list_total_regret.append(total_regrets)
            #list_total_reward_treatment.append(treatment_rewards)
            list_total_reward_spillover.append(spillover_rewards/treatment_rewards)
            list_total_reward_action_ratio.append(node_outcomes.count(1)/period)
            list_period.append(period)
            list_right_arm.append(similar_arm_context / (similar_arm_context + opposite_arm_context) * 100)
      print("Iteration: ", iter + 1, " Regret: ", total_regrets)
      list_list_total_reward_action_ratio.append(list_total_reward_action_ratio)
      list_list_total_reward.append(list_total_reward)
      list_list_total_regret.append(list_total_regret)
      list_list_total_reward_treatment.append(list_total_reward_treatment)
      list_list_total_reward_spillover.append(list_total_reward_spillover)
      list_list_period.append(list_period)
      list_list_right_arm.append(list_right_arm)
      list_list_changing_arm.append(changing_arm)

    RA_ratio.append(list_list_total_reward_action_ratio)
    Reward.append(list_list_total_reward)
    Regret.append(list_list_total_regret)
    Reward_treatment.append(list_list_total_reward_treatment)
    Reward_spillover.append(list_list_total_reward_spillover)
    Right_arm.append(list_list_right_arm)
    Period.append(list_list_period)
    Changing_Arms.append(list_list_changing_arm)
  Changing_Arms_avg.append(Changing_Arms)
  regres_avg.append(total_regrets)
  Reward_3D.append(Reward)
  Regret_3D.append(Regret)
  Reward_treatment_3D.append(Reward_treatment)
  Reward_spillover_3D.append(Reward_spillover)
  Right_arm_3D.append(Right_arm)
  Period_3D.append(Period)


#print(np.mean(Changing_Arms_avg, axis = 0), np.mean(regres_avg, axis = 0))

Outputs = [Period_3D, Reward_3D, Right_arm_3D, Reward_treatment_3D, Reward_spillover_3D, Regret_3D]
filename = method + "_" + algo + "_" + dataset
np.save(open(filename, 'wb') , Outputs)

