#@title decrease homophily
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

dataset = "pubmed" 
target_homophily = 0.3  # set the target homophily; for pubmed lowest is 0.30 and for hateful lowest is 0.60; dependent on the dataset
decay_granularity_param = 1000 # controls how much edges the method tests in each iteration

if dataset == "flickr":
  total_feature_count = 12047
  total_samples_count = 7575
  total_labels_count = 9
elif dataset == "pubmed":
  total_feature_count = 500
  total_samples_count = 19717
  total_labels_count = 3
elif dataset == "blogcatalog":
  total_feature_count = 8189
  total_samples_count = 5196
  total_labels_count = 6
elif dataset == "hateful":
  total_feature_count = 1036
  total_samples_count = 3218
  total_labels_count = 2
  
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

G = nx.Graph()
G = nx.from_pandas_edgelist(df_edges, source='source', target='target')

def homo_score():
  total_edges = len(df_edges)
  count = 0
  for index in range(len(df_edges)):
    v_i = df_edges.iloc[index, 0]
    v_j = df_edges.iloc[index, 1]
    if y_LR[v_i] == y_LR[v_j]:
      count += 1
  # print(count)
  score = count / total_edges
  # print("score:", score)
  return score

original_homophily = homo_score()
print("original homophilic score: ", original_homophily)  

#swapping by random to decrease homophily
while homo_score() > target_homophily:
  # print(homo_score())
  for k in range(decay_granularity_param):
    label_first = -1
    label_second = -1
    while label_first == label_second:
      label_first = random.randint(1, total_labels_count)
      label_second = random.randint(1, total_labels_count)

    #swapping between label_first and label_second
    ego_arm = None
    index = None
    while ego_arm != label_first: #ego label_first
      index = random.randint(0, total_samples_count - 1) #node_ID[index]
      ego_arm = y_LR[index]
    node_neighbours = list(G.neighbors(index))
    degree_label_first = 0
    degree_label_second = 0
    for j in node_neighbours:
      if (y_LR[j] == label_first):
        degree_label_first += 1
      elif (y_LR[j] == label_second):
        degree_label_second += 1

    _ego_arm = None
    _index = None
    while _ego_arm != label_second: #ego label_second
      _index = random.randint(0, total_samples_count - 1) #node_ID[_index]
      _ego_arm = y_LR[_index]
    node_neighbours = list(G.neighbors(_index))
    _degree_label_first = 0
    _degree_label_second = 0
    for j in node_neighbours:
      if (y_LR[j] == label_first):
        _degree_label_first += 1
      elif (y_LR[j] == label_second):
        _degree_label_second += 1

    if (degree_label_first - degree_label_second) > 0 and (_degree_label_second - _degree_label_first) > 0: #decrease homophily
      y_LR[index] = label_second
      y_LR[_index] = label_first
      temp = df_nodes.iloc[index].copy()
      df_nodes.iloc[index] = df_nodes.iloc[_index].copy()
      df_nodes.iloc[_index] = temp

target_homophily = homo_score()
print("Updated homophilic score: ", target_homophily)  
Output_features = df_nodes.to_numpy()
filename_features = f"df_nodes_{dataset}_{target_homophily:.2f}"
np.save(open(filename_features, 'wb') , Output_features)

Output_labels = y_LR
filename_labels = f"y_LR_{dataset}_{target_homophily:.2f}"
np.save(open(filename_labels, 'wb') , Output_labels)


