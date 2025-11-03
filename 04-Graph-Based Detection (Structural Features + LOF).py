import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# 1) Build / load a graph 
# Example synthetic graph: two communities + a few weird connectors
G = nx.barbell_graph(m1=40, m2=2)  
# add a few odd edges and star-like nodes to simulate anomalies
G.add_node("spoke_hub")
for i in range(10):
    G.add_edge("spoke_hub", f"leaf_{i}")
G.add_edge("spoke_hub", 5)
G.add_edge("spoke_hub", 60)

# 2) Compute structural features 
nodes = list(G.nodes())
deg = dict(G.degree())
clust = nx.clustering(G)
pr = nx.pagerank(G, alpha=0.85)

# Egonet features: for each node, subgraph of node+neighbors
def egonet_stats(G, u):
    ego = nx.ego_graph(G, u, radius=1)
    n = ego.number_of_nodes()
    m = ego.number_of_edges()
    # remove u to count edges among neighbors only
    ego_wo_u = ego.copy()
    ego_wo_u.remove_node(u)
    m_neighbors = ego_wo_u.number_of_edges()
    return n, m, m_neighbors

EGO = {u: egonet_stats(G, u) for u in nodes}

X = []
for u in nodes:
    n_ego, m_ego, m_nei = EGO[u]
    X.append([
        deg[u],
        clust[u],
        pr[u],
        n_ego,
        m_ego,
        m_nei,
        (m_ego - deg[u])  
    ])

X = np.asarray(X, dtype=float)

# 3) LOF on node features
scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X_s)                          
lof_score = -lof.negative_outlier_factor_              
rank = np.argsort(-lof_score)

top_k = 10
print("\n")
print("04-Graph-Based Detection (Structural Features + LOF)")
print("Top anomalous nodes:")
for i in rank[:top_k]:
    print(nodes[i], " | LOF score:", lof_score[i])
    
print("\n")
