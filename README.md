# NetCB: Network Contextual Bandits
In this repository, we provide implementation of baseline $CMAB$, $NetCB_{CMAB}$, $NetCB_{\overline{CMAB}}$, where `CMAB ∈ {LinUCB, NeuralUCB, NeuralTS, EENet, GNB}`. When the $CMAB$ is $LinUCB$, we denote the corresponding python files with LinUCB.py, NetCB-LinUCB.py, and NetCB-complete-LinUCB.py, respectively. 
# Run
Run $LinUCB$, $NetCB_{LinUCB}$, and $NetCB_{\overline{LinUCB}}$ on Blogcatalog dataset (homophilic score: 0.40) as follows:
```python
python bandit-experiments-real-world-datasets/blogcatalog_0.40/LinUCB.py
python bandit-experiments-real-world-datasets/blogcatalog_0.40/NetCB-LinUCB.py
python bandit-experiments-real-world-datasets/blogcatalog_0.40/NetCB-complete-LinUCB.py
```
Run $LinUCB$, $NetCB_{LinUCB}$, and $NetCB_{\overline{LinUCB}}$ on semi-synthetic Blogcatalog dataset (homophilic score: 0.88):
```python
python bandit-experiments-semi-synthetic-datasets/blogcatalog_0.88/LinUCB.py
python bandit-experiments-semi-synthetic-datasets/blogcatalog_0.88/NetCB-LinUCB.py
python bandit-experiments-semi-synthetic-datasets/blogcatalog_0.88/NetCB-complete-LinUCB.py
```

## Prerequisites: 
python 3.9.7, CUDA 11.6, torch 2.0.1 (for `CMAB ∈ {EENet, GNB}`), torch 1.12.1+cu113 (for `CMAB ∈ {NeuralUCB, NeuralTS}`), torchvision 0.16.2, sklearn 0.24.2, numpy 1.20.3, scipy 1.7.1, pandas 1.3.4



