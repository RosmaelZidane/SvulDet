import os
import torch
import pandas as pd
from pytorch_lightning import Trainer
from dgl.dataloading import GraphDataLoader
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f1model2 import LitMultiTaskGAT, GraphFunctionDataset  
import uutils.__utils__ as utls
from processing.dataprocessing import dataset  

def evaluate_best_model(checkpoint_path, config):
    # Load dataset
    df = dataset()
    graph_dir = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
    test_set = GraphFunctionDataset(df, graph_dir, split='test')
    test_loader = GraphDataLoader(test_set, batch_size=1, shuffle=False, num_workers=torch.get_num_threads())

    # Load the model from checkpoint
    model = LitMultiTaskGAT.load_from_checkpoint(checkpoint_path, config=config)

    # Run test
    trainer = Trainer(logger=False)
    trainer.test(model, test_loader)

    print("Evaluation complete.")

# if __name__ == '__main__':
#     # Fill in the path to the best checkpoint and the config used to train it
#     checkpoint_path = os.path.join(utls.cache_dir(), "checkpoints", "trial_3.ckpt")  # Change this as needed
#     best_config = {
#         'in_feats':  768,
#         'hidden_feats': 256,
#         'num_heads': 4,
#         'dropout': 0.2,
#         'lr': 1e-4,
#     }

#     evaluate_best_model(checkpoint_path, best_config)


print(dir(utls))
