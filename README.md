# NetCB: Network Contextual Bandits
In this repository, we provide implementation of baselines $CMAB$s, $NetCB_{CMAB}$, $NetCB_{\overline{CMAB}}$, where $CMAB \in {LinUCB, NeuralUCB, NeuralTS, EENet, GNB}$. When the $CMAB$ is $LinUCB$, we denote the corresponding python files with LinUCB.py, NetCB-LinUCB.py, and NetCB-complete-LinUCB.py, respectively. 
# Run
Run LinUCB on Blogcatalog dataset (homophilic score: 0.40):
python bandit-experiments-real-world-datasets/blogcatalog_0.40/LinUCB.py
