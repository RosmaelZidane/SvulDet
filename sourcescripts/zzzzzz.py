from dgl import load_graphs


def print_graph_data(g):
    """Print all node and edge data names from a DGL graph."""
    print("Node Data:")
    for key in g.ndata:
        print(f" - {key}: {g.ndata[key].shape}")

    print("\nEdge Data:")
    for key in g.edata:
        print(f" - {key}: {g.edata[key].shape}")

paths = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/Graph/dataset_svuldet_codebert_pdg+raw/5503"

gcb = load_graphs(paths)[0][0]


print_graph_data(gcb)


print(gcb.number_of_nodes())




print(gcb.ndata["_CODEBERT"])


