import os
import torch as th
import torch.nn as nn
import torch
import dgl
from dgl.nn import SAGEConv
import networkx as nx
from dgl import load_graphs
from node2vec import Node2Vec
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from GATmain import  GraphFunctionDataset

def compute_graph_embedding(graph, method='mean'):
    """--- Graph Embedding Computation --"""
    h = graph.ndata['_FUNC_EMB'].float()

    if method == 'mean':
        return h.mean(dim=0)

    elif method == 'max':
        return h.max(dim=0)[0]

    elif method == "Node2Vec":
        #print("[Infos] Method graph to vector: Node2Vec")
        def node2vecmodel(graph):
            nxg = graph.to_networkx()
            node2vecmodel = Node2Vec(nxg, dimensions=128, walk_length=5, num_walks=10, workers=1)
            node2vec_fitted = node2vecmodel.fit(window=5, min_count=1, batch_words=2)
            embeddings = node2vec_fitted.wv
            node_embeddings = {int(node): embeddings[str(node)] for node in nxg.nodes}
            embedding_matrix = th.tensor([node_embeddings[node.item()] for node in graph.nodes()], dtype=th.float)
            return embedding_matrix.mean(dim=0)
        return node2vecmodel(graph)

    elif method == "GraphSAGE":
        #print("[Infos] Method graph to vector: GraphSAGE")
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


def build_dependency_graph_from_labels_and_similarity(embeddings, labels, similarity_threshold=0.06):
    """# -- Dependency Graph Construction --"""
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

def create_function_level_dependency_graph(embeddings, method, eps=0.6, min_samples=2, 
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
        raise ValueError(f"Unsupported clustering method: {method}. Use 'dbscan' or 'kmeans'.")

    dependency_graph = build_dependency_graph_from_labels_and_similarity(embeddings, 
                                                                         labels, similarity_threshold)
    return dependency_graph, labels


def networkx_to_dgl(nx_graph, feature_size, device='cpu'):
    """
    Converts a NetworkX graph to a DGLGraph and assigns random node features.
    """
    dgl_graph = dgl.from_networkx(nx_graph)

    dgl_graph = dgl_graph.to(device)

    num_nodes = dgl_graph.num_nodes()

    random_features = torch.randn(num_nodes, feature_size, device=device)
    dgl_graph.ndata['_FUNC_EMB'] = random_features

    return dgl_graph


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

###


# # Dependecy graph contruction
# class GetlistGL:
#     def __init__(self, df, config_grid, split='train'):
#         self.param = config_grid['embedd_method']
#         self.graph_dir = self.graph_path()
#         self.dataset = GraphFunctionDataset(df, self.graph_dir, split)
#         self.graph_list = []

#     def get_graph_list(self):
#         if not self.graph_list:  # Only load if not already loaded
#             for idx in range(len(self.dataset)):
#                 g = self.dataset[idx]
#                 self.graph_list.append(g)
#         return self.graph_list

#     def graph_path(self):
#         if self.param == "Codebert":
#             return f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
#         elif self.param == "Word2vec":
#             return f"{utls.cache_dir()}/Graph/dataset_svuldet_word2vec_pdg+raw" 
#         elif self.param == "Sbert":
#             return f"{utls.cache_dir()}/Graph/dataset_svuldet_sbert_pdg+raw" 
#         else: 
#             raise ValueError("[Error] Provide a valid embedding model name: 'Codebert', 'Sbert', or 'Word2vec'")

#     @staticmethod
#     def dependency_graph_construction(graph_list, config_grid):
#         graph_embeddings = [
#             compute_graph_embedding(g, method=config_grid['graph_to_vec_method']).numpy()
#             for g in graph_list
#         ]
#         function_dependency_graph, cluster_labels = create_function_level_dependency_graph(
#             embeddings=graph_embeddings,
#             method=config_grid['cluster_method'],
#             eps=0.8,
#             min_samples=2,
#             n_clusters=2,
#             similarity_threshold=config_grid['cos_sim_threshold']
#         )
#         return function_dependency_graph, cluster_labels










# ----------- Main Execution -----------

# gpath = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/Graph/dataset_svuldet_codebert_pdg+raw/3"
# gpath2 = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/Graph/dataset_svuldet_codebert_pdg+raw/5"
# gpath3 = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/Graph/dataset_svuldet_codebert_pdg+raw/50"
# graph = load_graphs(gpath)[0][0]
# graph2 = load_graphs(gpath2)[0][0]
# graph3 = load_graphs(gpath3)[0][0]

# # Collect embeddings
# graphs_list = [graph, graph2, graph3]


# graph_embeddings = [compute_graph_embedding(g, method='Node2Vec').numpy() for g in graphs_list]

# chosen_method = "kmeans" #"dbscan" #   "kmeans"  # or "dbscan"
# similarity_threshold = 0.06  # ~6% cosine similarity

# function_dependency_graph, cluster_labels = create_function_level_dependency_graph(
#     embeddings=graph_embeddings,
#     method=chosen_method,
#     eps=0.8,
#     min_samples=2,
#     n_clusters=2,
#     similarity_threshold=similarity_threshold
# )

# # visualize_dependency_graph(function_dependency_graph, filename=f"dependency_graph_{chosen_method}.png")

# print("\nFinal Function-Level Dependency Graph:")
# print(f"Nodes: {list(function_dependency_graph.nodes())}")
# print(f"Edges: {list(function_dependency_graph.edges())}")
# print(f"Cluster labels: {cluster_labels}")


