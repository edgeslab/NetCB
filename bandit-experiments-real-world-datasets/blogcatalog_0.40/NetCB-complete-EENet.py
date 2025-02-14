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
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from scipy.stats import linregress
import torchvision
from skimage.measure import block_reduce

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

algo = "EENet"

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
else:
  features = pickle.load(open('attrs.pkl', 'rb'),  encoding='latin1')
  df_nodes = pd.DataFrame(features.toarray())
  df_edges = pd.read_csv('edgelist.txt', sep="\s+", header=None, names=["target", "source"])
  df_l = pd.read_csv('labels.txt', sep="\s+", header=None, names=["ID", "label"])
  y_LR = df_l.to_numpy()[:, 1]

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
alpha = 2
node_outcomes = [-1] * total_samples_count
flag_id = [-1] * total_samples_count          
treatment_id = [-1] * total_samples_count  
arm_type = [-1] * total_samples_count        
predicted_arm = [-1] * total_samples_count   

Aa = None
Aa_inv = None
ba = None
theta = None
max_a = None
x = None
highest_spillover = None
lowest_spillover = None


'''Network Structure'''

class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class Network_exploration(nn.Module):
    def __init__(self, input_dim, _kernel_size = 100,  _stride = 50,  _channels = 1):
        super(Network_exploration, self).__init__()
        self.conv1 = nn.Conv1d(1, _channels, kernel_size = _kernel_size, stride = _stride)
        _num_dim = int(((input_dim - _kernel_size)/ _stride + 1) * _channels)
        self.fc1 = nn.Linear(_num_dim, 100)
        self.fc2 = nn.Linear(100, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.activate(self.fc1(x))
        x = self.fc2(x)
        return x

class Network_decision_maker(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_decision_maker, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))



'''Network functions'''

class Exploitation:
    def __init__(self, input_dim, num_arm, pool_step_size, lr = 0.01, hidden=100):
        '''input_dim: number of dimensions of input'''
        '''num_arm: number of arms'''
        '''lr: learning rate'''
        '''hidden: number of hidden nodes'''

        self.func = Network_exploitation(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr
        self.pool_step_size = pool_step_size
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)


    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)


    def output_and_gradient(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        results = self.func(tensor)
        g_list = []
        res_list = []
        for fx in results:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(np.array(g.cpu()))
            res_list.append([fx.item()])
        res_list = np.array(res_list)
        g_list = np.array(g_list)

        #'''Gradient Aggregation'''
        g_list = block_reduce(g_list, block_size=(1, self.pool_step_size), func=np.mean)
        return res_list, g_list

    def return_gradient(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        result = self.func(tensor)
        self.func.zero_grad()
        result.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
        g = [np.array(g.cpu())]
        g = np.array(g)
        g_aggre = block_reduce(g, block_size=(1, self.pool_step_size), func=np.mean)
        return result.item(), g_aggre[0]

    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                loss = (self.func(c.to(device)) - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length




class Exploration:
    def __init__(self, input_dim, lr=0.001):
        self.func = Network_exploration(input_dim=input_dim).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr


    def update(self, context, reward):
        tensor = torch.from_numpy(context).float().to(device)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.unsqueeze(tensor, 0)
        self.context_list.append(tensor)
        self.reward.append(reward)

    def output(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        tensor = torch.unsqueeze(tensor, 1)

        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return res


    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr, weight_decay=0.0001)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                output = self.func(c.to(device))
                optimizer.zero_grad()
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / cnt
            if batch_loss / length <= 2e-3:
                #print("batched loss",  batch_loss / length)
                return batch_loss / length




class Decision_maker:
    def __init__(self, input_dim, hidden=20, lr = 0.01):
        self.func = Network_decision_maker(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        # print("f3_lr", self.lr)


    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)


    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return np.argmax(res)

    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                target = torch.tensor([r]).unsqueeze(1).to(device)
                output = self.func(c.to(device))
                loss = (output - r)**2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length

class EE_Net:
    def __init__(self, low, high, low_activation, high_activation, dim, n_arm, pool_step_size, lr_1 = 0.01, lr_2 = 0.01, lr_3 = 0.01, hidden=100, neural_decision_maker = False):
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


        #Network 1
        self.f_1 = Exploitation(dim, n_arm, pool_step_size, lr_1, hidden)

        # number of dimensions of aggregated for f_2
        f_2_input_dim = self.f_1.total_param // pool_step_size + 1
        #Network 2
        self.f_2 = Exploration(f_2_input_dim, lr_2)

        #Network 3
        self.f_3 = Decision_maker(2, 20, lr_3)

        self.arm_select = 0

        self.exploit_scores = []
        self.explore_scores = []
        self.ee_scores = []
        self.grad_list = []

        self.contexts = []
        self.rewards = []
        self.decision_maker = neural_decision_maker

    def predict(self, context, t):  
        self.exploit_scores, self.grad_list = self.f_1.output_and_gradient(context)
        self.explore_scores = self.f_2.output(self.grad_list)
        self.ee_scores = np.concatenate((self.exploit_scores, self.explore_scores), axis=1)

        if self.decision_maker and t > 500:
            # neural decision maker
            self.arm_select = self.f_3.select(self.ee_scores)
        else:
            # linear decision maker
            f_2_weight = 1.0
            if t > 1000: f_2_weight = 0.01
            suml = self.exploit_scores + f_2_weight * (self.explore_scores-1.0)
            self.arm_select = np.argmax(suml)
        return self.arm_select

    def update(self, context, r_1, t):
        # update exploitation network
        self.f_1.update(context[self.arm_select], r_1)

        self.contexts.append(context[self.arm_select])
        self.rewards.append(r_1)

        # update exploration network
        f_1_predict = self.exploit_scores[self.arm_select][0]
        r_2 = (r_1 - f_1_predict) + 1.0
        self.f_2.update(self.grad_list[self.arm_select], r_2)

        # add additional samples to exploration net when the selected arm is not optimal
        if t < 1000:
            if r_1 == 0:
                index = 0
                for i in self.grad_list:
                    c = 1.2
                    if index != self.arm_select:
                        self.f_2.update(i, c)
                    index += 1

        # update decision maker
        self.f_3.update(self.ee_scores[self.arm_select], float(r_1))

    def train(self, t):
        #train networks
        loss_1 = self.f_1.train()
        loss_2 = self.f_2.train()
        if self.decision_maker:
            loss_3 = self.f_3.train()
        else:
            loss_3 = 0.0
        return loss_1, loss_2, loss_3

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
          if (node_outcomes[j] == 0)  and (label == model.predict(format_features(df_nodes.iloc[j], total_labels_count, total_feature_count * total_labels_count, total_feature_count), freq)):  
            m += 1
        reward_predicted_arm = DTE_right + DTE_right * e_ij_p_high * m 
        expected_reward_arms[label] = reward_predicted_arm
      else:
        #where m_prime is the total number of inactive neighbors whose predicted arms are the opposite to that of predicted arm of the arrival node
        node_neighbours = list(G.neighbors(id)) 
        m_prime = 0
        for j in node_neighbours:   
          if (node_outcomes[j] == 0) and (label == model.predict(format_features(df_nodes.iloc[j], total_labels_count, total_feature_count * total_labels_count, total_feature_count), freq)):
            m_prime += 1
        reward_alternate_arm = DTE_wrong + DTE_wrong * e_ij_p_low * m_prime
        expected_reward_arms[label] = reward_alternate_arm

    highest_reward_label= np.argmax(expected_reward_arms)
    reward_alternate_arm = expected_reward_arms[highest_reward_label]


    if (reward_alternate_arm - reward_predicted_arm > against_threshold):
      arm = highest_reward_label  
      # print("changing the arm") 
      changing_arm += 1
      is_arm_changed = True

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

def format_features(user_features, n_arm, dim, total_feature_count):
  X = np.zeros((n_arm, dim))
  for a in range(n_arm):
      X[a, a * total_feature_count:a * total_feature_count +
          total_feature_count] = user_features     
  return X

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
  against_diff = []
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


      ee_net = EE_Net(l_spillover , h_spillover, incorrect_activation, correct_activation, total_feature_count * total_labels_count, total_labels_count, pool_step_size = 50, lr_1 = 0.001, lr_2 = 0.001, lr_3 = 0.001,  hidden=100, neural_decision_maker = False)
      #print(0.001, 0.001, 0.001)
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
          context = format_features(user_features, total_labels_count, total_feature_count * total_labels_count, total_feature_count)
          arm_select = ee_net.predict(context, period)
          r, is_arm_changed,regrets = generate_outcome(arm_select, i, spillover_index, period, ee_net)
          total_regrets += regrets

          ee_net.update(context, r, period)

          period += 1
          # print(context, arm_select, r)

          if period<1000 and is_arm_changed == False:
              if period%10 == 0:
                  loss_1, loss_2, loss_3  = ee_net.train(period)
          else:
              if period%100 == 0 and is_arm_changed == False:
                  loss_1, loss_2, loss_3  = ee_net.train(period)
           
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