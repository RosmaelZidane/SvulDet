import os
import torch as th
import torch.nn as nn
import dgl
from dgl.nn import SAGEConv
import networkx as nx
from dgl import load_graphs
from node2vec import Node2Vec
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ----------- Graph Embedding Computation -----------

def compute_graph_embedding(graph, method='mean'):
    h = graph.ndata['_FUNC_EMB'].float()

    if method == 'mean':
        return h.mean(dim=0)

    elif method == 'max':
        return h.max(dim=0)[0]

    elif method == "Node2Vec":
        print("[Infos] Method graph to vector: Node2Vec")
        def node2vecmodel(graph):
            nxg = graph.to_networkx()
            node2vecmodel = Node2Vec(nxg, dimensions=128, walk_length=5, num_walks=10, workers=4)
            node2vec_fitted = node2vecmodel.fit(window=5, min_count=1, batch_words=2)
            embeddings = node2vec_fitted.wv
            node_embeddings = {int(node): embeddings[str(node)] for node in nxg.nodes}
            embedding_matrix = th.tensor([node_embeddings[node.item()] for node in graph.nodes()], dtype=th.float)
            return embedding_matrix.mean(dim=0)
        return node2vecmodel(graph)

    elif method == "GraphSAGE":
        print("[Infos] Method graph to vector: GraphSAGE")
        class GraphSAGE(nn.Module):
            def __init__(self, in_feats, hidden_feats, out_feats):
                super(GraphSAGE, self).__init__()
                self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
                self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator_type='mean')

            def forward(self, g, inputs):
                h = self.conv1(g, inputs)
                h = th.relu(h)
                h = self.conv2(g, h)
                return h

        def graphsage_model(graph):
            node_feats = graph.ndata['_FUNC_EMB'].float()
            in_feats = node_feats.shape[1]
            model = GraphSAGE(in_feats=in_feats, hidden_feats=64, out_feats=128)
            model.eval()
            with th.no_grad():
                node_embeddings = model(graph, node_feats)
            return node_embeddings.mean(dim=0)

        return graphsage_model(graph)

    else:
        raise ValueError(f"Unknown method: {method}")

# ----------- Dependency Graph Construction -----------

def build_dependency_graph_from_labels_and_similarity(embeddings, labels, similarity_threshold=0.06):
    graph = nx.Graph()
    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == -1 or labels[j] == -1:
                continue
            if labels[i] == labels[j] and similarity_matrix[i][j] >= similarity_threshold:
                graph.add_edge(i, j)
    return graph

def create_function_level_dependency_graph(embeddings, method="dbscan", eps=0.8, min_samples=2, 
                                           n_clusters=2, similarity_threshold=0.06):
    if method.lower() == "dbscan":
        print("[Infos] Clustering method: DBSCAN")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)

    elif method.lower() == "kmeans":
        print("[Infos] Clustering method: KMeans")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)

    else:
        raise ValueError("Unsupported clustering method. Use 'dbscan' or 'kmeans'.")

    dependency_graph = build_dependency_graph_from_labels_and_similarity(embeddings, labels, similarity_threshold)
    return dependency_graph, labels

# ----------- Visualization -----------

def visualize_dependency_graph(dependency_graph, title="Function-Level Dependency Graph", filename="dependency_graph.png"):
    pos = nx.spring_layout(dependency_graph, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(
        dependency_graph,
        pos,
        with_labels=True,
        node_color='skyblue',
        node_size=800,
        edge_color='gray'
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Saved] {filename}")




# ----------- Main Execution -----------

gpath = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/Graph/dataset_svuldet_codebert_pdg+raw/3"
gpath2 = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/Graph/dataset_svuldet_codebert_pdg+raw/5"
graph = load_graphs(gpath)[0][0]
graph2 = load_graphs(gpath2)[0][0]

graphs_list = [graph, graph2]
graph_embeddings = [compute_graph_embedding(g, method='Node2Vec').numpy() for g in graphs_list]

chosen_method = "dbscan" #   "kmeans"  # or "dbscan"
similarity_threshold = 0.06  # ~6% cosine similarity

function_dependency_graph, cluster_labels = create_function_level_dependency_graph(
    embeddings=graph_embeddings,
    method=chosen_method,
    eps=0.8,
    min_samples=2,
    n_clusters=2,
    similarity_threshold=similarity_threshold
)

# visualize_dependency_graph(function_dependency_graph, filename=f"dependency_graph_{chosen_method}.png")

print("\nFinal Function-Level Dependency Graph:")
print(f"Nodes: {list(function_dependency_graph.nodes())}")
print(f"Edges: {list(function_dependency_graph.edges())}")
print(f"Cluster labels: {cluster_labels}")
