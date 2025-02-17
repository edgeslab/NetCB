# import networkx as nx
import numpy as np
from numpy import *
# from sklearn.metrics import jaccard_similarity_score
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from operator import itemgetter
from copy import deepcopy
#import mcl
# import drawing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import sklearn.metrics
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
# import markov_clustering as mc
import matplotlib.pyplot as mp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import random
import math

total_feature_count = 1036
total_samples_count = 4971

df_edges = pd.read_csv('edges.csv', header=None, names=["target", "source"])

for i in range(len(df_edges)):
  df_edges.iloc[i, 0] = str(df_edges.iloc[i, 0])
  df_edges.iloc[i, 1] = str(df_edges.iloc[i, 1])

feature_names = ["w_{}".format(ii) for ii in range(total_feature_count)]
column_names =  feature_names + ["subject"]
df_nodes = pd.read_csv('features.csv', header=None, names=column_names)

df_type = pd.read_csv('nodes.csv', header=None, names=column_names)
existing_nodes = list(df_type.iloc[0:, 0])

R = []
for i in range(len(df_nodes)):
  if df_nodes.iloc[i, 0] not in existing_nodes:
    R.append(i)
R = list(set(R))
df_nodes = df_nodes.drop(R)

df_nodes.index = df_nodes.iloc[0:, 0]

total_samples_count = 3218
#start

all_node_list = list(df_nodes.index[:total_samples_count])
for i in range(len(all_node_list)):
  all_node_list[i] = str(all_node_list[i])

R = []
for i in range(len(df_edges)):
  if (df_edges.iloc[i, 0] not in all_node_list) or (df_edges.iloc[i, 1] not in all_node_list):
    R.append(i)

R = list(set(R))
df_edges = df_edges.drop(R)

G = nx.Graph()
G = nx.from_pandas_edgelist(df_edges, source='source', target='target')

df_nodes = df_nodes.drop(["w_0"], axis=1)  #dropping last column for now


# df_type = pd.read_csv('nodes.csv', header=None, names=column_names)
all_type_list = list(df_type.iloc[0:, 1])


y_LR = []
for i in range(len(all_type_list)):
  if all_type_list[i] == -1:
    y_LR.append(1)
  elif all_type_list[i] == 1:
    y_LR.append(2)

probability_treatment_node = None
probability_control_node = None
total_feature_count = 1036

I = (df_nodes.index).to_numpy()
J = []
for i in range(len(I)):
  J.append(str(I[i]))
  
E = []
for i in range(len(df_edges)):
  E.append([J.index(df_edges.iloc[i, 0]), J.index(df_edges.iloc[i, 1])])

F = []
for i in range(len(df_nodes)):
  F.append((df_nodes.iloc[i]).to_numpy())
  
np.save(open("labels", 'wb') , y_LR)
np.save(open("features", 'wb') , F)
np.save(open("edges", 'wb') , E)

