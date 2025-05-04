

#  mat1 and mat2 shapes cannot be multiplied (18x768 and 100x256)

import os       
import numpy as np          


def get_or_compute_v1(df, config_grid, 
                      save_path=f"/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/v1.npy"):
    """
    Computes v1 if not already saved; otherwise, loads it from disk.
    """
    def replace_nan_with_zeros(vec):
        if np.isnan(vec).any():
            vec = np.zeros_like(vec)
            return vec
    if os.path.exists(save_path):
        print(f"Loading v1 from {save_path}...")
        v1 = np.load(save_path)
    else:
        print(f"v1 not found. Computing and saving to {save_path}...")
        gl_loader = GetlistGL(df, config_grid, split='train')
        graph_list = gl_loader.get_graph_list()
        gl, _ = GetlistGL.dependency_graph_construction(graph_list, config_grid)
        
        glnx = GLs.networkx_to_dgl(
            nx_graph=gl,
            feature_size=config_grid['gl_vec_length'],
            device='cpu'
        )
        v1 = GLs.compute_graph_embedding(
            glnx,
            method=config_grid['graph_to_vec_method']
        ).cpu().numpy()
        v1 = replace_nan_with_zeros(v1)
        np.save(save_path, v1)
    return v1



df = None
config_grid = {}
v1 = get_or_compute_v1(df, config_grid)

print(len(v1))