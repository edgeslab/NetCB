#@title increase homophily

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

dataset = "blogcatalog" 

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
  
N_count = [0]*len(df_nodes)
for index in range(len(df_nodes)):
  N_count[index] = len(list(G.neighbors(index)))
  
# remove edges by random to increase homophily
# need to repeat the process if more higher homophily is needed than the method achieves
count = 0
indices = []
for index in range(len(df_edges)):
  v_i = df_edges.iloc[index, 0]
  v_j = df_edges.iloc[index, 1]
  if y_LR[v_i] != y_LR[v_j] and N_count[v_i] > 1 and N_count[v_j] > 1:
    indices.append(index)
    N_count[v_i] -= 1
    N_count[v_j] -= 1
    count += 1

print("Number of potential edges to be removed: ", len(indices))    
remove_n = 149000  #controls how much you want to increase homophily, which should be smaller than the total number of edges to be removed
random.shuffle(indices)
drop_indices = random.sample(indices, remove_n)
df_edges = df_edges.drop(drop_indices)
df_edges.index = list(range(0, len(df_edges.index)))

target_homophily = homo_score()
print("Updated homophilic score: ", target_homophily)  
Output_edges = df_edges.to_numpy()
filename_edges = f"df_edges_{target_homophily:.2f}"
np.save(open(filename_edges, 'wb') , Output_edges)

