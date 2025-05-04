import os
import torch
import torch.nn.functional as F
import numpy as np
import dgl
from dgl import DGLGraph
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def compute_graph_embedding(graph, method='mean'):
    """Compute a vector representation of the graph by aggregating node features.
    some possibility are the use of method such as mean, node2vec, Graph2Vec"""
    h = graph.ndata['_FUNC_EMB'].float()  # Or use another embedding
    if method == 'mean':
        return h.mean(dim=0).cpu().numpy()
    elif method == 'max':
        return h.max(dim=0)[0].cpu().numpy()
    
    else:
        raise ValueError("Unknown method")

def build_dependency_graph(graphs, eps=0.5, min_samples=2):
    """
    Create a graph of subgraphs. Each original graph is represented by a node.
    Clustering similar function graphs and adding cluster hub nodes.
    """
    embeddings = [compute_graph_embedding(g) for g in graphs]
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_embeddings)
    labels = clustering.labels_

    cluster_graph = dgl.graph([])
    num_graphs = len(graphs)

    # Create nodes: One for each function graph + one for each cluster
    cluster_ids = sorted(set(labels))
    num_clusters = len([c for c in cluster_ids if c != -1])
    cluster_graph.add_nodes(num_graphs + num_clusters)

    # Map graph index to node and cluster
    graph_nodes = list(range(num_graphs))
    cluster_nodes = {c: num_graphs + i for i, c in enumerate(cluster_ids) if c != -1}

    edges_src = []
    edges_dst = []

    for i, label in enumerate(labels):
        if label != -1:
            edges_src.append(i)  # function node
            edges_dst.append(cluster_nodes[label])  # cluster hub
            edges_src.append(cluster_nodes[label])
            edges_dst.append(i)

    cluster_graph.add_edges(edges_src, edges_dst)
    cluster_graph.ndata['type'] = torch.tensor([0]*num_graphs + [1]*num_clusters)  # 0: func, 1: cluster
    cluster_graph.ndata['emb'] = torch.tensor(np.vstack([scaled_embeddings] + [np.zeros_like(scaled_embeddings[0])] * num_clusters)).float()
    return cluster_graph, labels

class ClusterAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_fc = torch.nn.Linear(embed_dim, 1)

    def forward(self, cluster_graph, func_indices):
        h = cluster_graph.ndata['emb']
        attn_weights = torch.sigmoid(self.attn_fc(h)).squeeze(-1)
        attention = attn_weights[func_indices]  # Use attention only for function nodes
        return attention

# During training, after loading the batch of function graphs:
# 1. Compute dependency graph from batch (you can move it to dataset if static)
# 2. Compute attention values from it
# 3. Use those attention weights as additional features or modulate existing attention layers in your GAT

# Add the attention value to node features
# g.ndata['_ATTN'] = attention_value_for_g
# And concatenate with original features during input feature construction
