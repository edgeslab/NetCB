# NetCB: Network Contextual Bandits
In this repository, we provide implementation of baseline $CMAB$, $NetCB_{CMAB}$, $NetCB_{\overline{CMAB}}$, where $CMAB \in \{LinUCB, NeuralUCB, NeuralTS, EENet, GNB\}$. When the $CMAB$ is $LinUCB$, we denote the corresponding python files with LinUCB.py, NetCB-LinUCB.py, and NetCB-complete-LinUCB.py, respectively. 
# Run
Run $LinUCB$, $NetCB_{LinUCB}$, and $NetCB_{\overline{CMAB}}$ on Blogcatalog dataset (homophilic score: 0.40) as follows:
```python
python bandit-experiments-real-world-datasets/blogcatalog_0.40/LinUCB.py
python bandit-experiments-real-world-datasets/blogcatalog_0.40/NetCB-LinUCB.py
python bandit-experiments-real-world-datasets/blogcatalog_0.40/NetCB-complete-LinUCB.py
```
Run LinUCB$, $NetCB_{LinUCB}$, and $NetCB_{\overline{CMAB}}$ on semi-synthetic Blogcatalog dataset (homophilic score: 0.88):
```python
python bandit-experiments-semi-synthetic-datasets/blogcatalog_0.88/LinUCB.py
python bandit-experiments-semi-synthetic-datasets/blogcatalog_0.40/NetCB-LinUCB.py
python bandit-experiments-semi-synthetic-datasets/blogcatalog_0.40/NetCB-LinUCB.py
```
