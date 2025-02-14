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
from scipy.sparse import csr_matrix
from tqdm import tqdm

method = "CMAB"
dataset = "blogcatalog" 
algo = "LinUCB"

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

probability_treatment_node = None
probability_control_node = None


node_IDs = np.load(open("node_IDs", 'rb'), allow_pickle = True)


similar_arm_context = 0
opposite_arm_context = 0

changing_arm = None

treatment_rewards = 0
spillover_rewards = 0

d = total_feature_count
alpha = hyper_param_1
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
n_arms = total_labels_count

def init_arms(_alpha, low, high, low_activation, high_activation, number_of_arms):
  global Aa
  global Aa_inv
  global ba
  global theta
  global node_outcomes
  global flag_id
  global treatment_id
  global arm_type
  global alpha
  global highest_spillover
  global lowest_spillover
  global probability_treatment_node
  global probability_control_node
  global predicted_arm
  global indices
  global n_arms 

  d = total_feature_count
  alpha = _alpha
  node_outcomes = [-1] * total_samples_count
  flag_id = [-1] * total_samples_count          
  treatment_id = [-1] * total_samples_count  
  arm_type = [-1] * total_samples_count        
  predicted_arm = [-1] * total_samples_count        
  highest_spillover = high
  lowest_spillover = low
  probability_treatment_node = high_activation
  probability_control_node = low_activation
  n_arms = number_of_arms
  
  Aa = np.zeros((n_arms, d, d))   
  Aa_inv = np.zeros((n_arms, d, d)) 
  ba = np.zeros((n_arms, d, 1)) 
  theta = np.zeros((n_arms, d, 1))  

  indices = []
  for i in range(n_arms):
    Aa[i] = np.identity(d)
    Aa_inv[i] = np.identity(d)
    ba[i] = np.zeros((d, 1))
    theta[i] = np.zeros((d, 1))
    indices.append(i)

def update(arm, id, spillover_index):
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
  
  optimal_rewards = max_rewards(id) 
  prev_node_count = node_outcomes.count(1) 

  context = y_LR[id]
  r = None
  flag = None

  predicted_arm[id] = arm

  if arm == context:
    r = np.array([1])
    flag = np.random.binomial(1, probability_treatment_node, 1)
    similar_arm_context += 1
    arm_type[id] = 1
  else:
    r = np.array([0])
    flag = np.random.binomial(1, probability_control_node, 1)
    opposite_arm_context += 1
    arm_type[id] = 0

  node_outcomes[id] = flag[0]
  flag_id[id] = arm   #assigned_arm
  treatment_id[id] = arm

  is_visited = [False] * total_samples_count
  is_visited[id] = True
  queue = []

  prev_treatment_rewards = treatment_rewards

  #contagion to neighbour nodes
  if node_outcomes[id] == 1:
    node_neighbours = list(G.neighbors(id)) 
    for j in node_neighbours:   
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
  
  after_node_count = node_outcomes.count(1) 
  after_treatment_rewards = treatment_rewards
  predicted_rewards = after_node_count - prev_node_count 
  

  r[0] = flag[0]

  Aa[arm] += np.outer(x,x)
  Aa[arm] += (0.0000001 * np.random.rand(total_feature_count, total_feature_count)) #avoid singular matrix inverse
  Aa_inv[arm] = np.linalg.inv(Aa[arm])
  ba[arm] += r * x
  theta[arm] = Aa_inv[arm].dot(ba[arm])
  
  if predicted_rewards > optimal_rewards:
    return 0
  else:
    return optimal_rewards - predicted_rewards 

def recommend(user_features):
  global max_a
  global x
  global indices

  x = np.array(user_features).reshape((d,1))
  x_t = np.transpose(x)
  UCB = np.matmul(np.transpose(theta[indices],(0,2,1)), x) + alpha * np.sqrt(np.matmul(x_t, Aa_inv[indices].dot(x)))

  max_index = np.argmax(UCB)
  max_a = max_index
  
  return max_index
  
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


Changing_Arms_avg = [] #3D array
Reward_3D = []
Regret_3D = []
Period_3D = []
Reward_treatment_3D = []
Reward_spillover_3D = []
Right_arm_3D = []

for iter in range(iteration_count):
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
      #print(l_spillover,h_spillover)
      similar_arm_context = 0
      opposite_arm_context = 0
      changing_arm = 0
      _alp = hyper_param_1

      node_outcomes = [-1] * total_samples_count
      flag_id = [-1] * total_samples_count          
      treatment_id = [-1] * total_samples_count
      arm_type = [-1] * total_samples_count 

      init_arms(_alp, l_spillover , h_spillover, incorrect_activation, correct_activation, total_labels_count)
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
        user_features = df_nodes.iloc[i]
        if node_outcomes[i] != 1:
          a = recommend(user_features)
          regrets = update(a, i, spillover_index)
          total_regrets += regrets
          period += 1
          if period % 10 == 0: 
            list_total_reward.append(node_outcomes.count(1))
            list_total_regret.append(total_regrets)
            list_total_reward_treatment.append(treatment_rewards)
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
  Reward_3D.append(Reward)
  Regret_3D.append(Regret)
  Reward_treatment_3D.append(Reward_treatment)
  Reward_spillover_3D.append(Reward_spillover)
  Right_arm_3D.append(Right_arm)
  Period_3D.append(Period)


avg_arms = np.mean(Changing_Arms_avg, axis = 0)


#print(avg_arms)

Outputs = [Period_3D, Reward_3D, Right_arm_3D, Reward_treatment_3D, Reward_spillover_3D, Regret_3D]  
filename = method + "_" + algo + "_" + dataset
np.save(open(filename, 'wb') , Outputs)

