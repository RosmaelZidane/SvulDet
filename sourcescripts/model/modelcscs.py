# Modify the code below so that it use all the three feature, but create a from the three g.ndata['_RANDFEAT'], g.ndata['_FUNC_EMB'], g.ndata['_CODEBERT'], create and reshape them to the same size as "_CODEBERT"
# use f1 score as a target metric chech instead of accucacy. if the metric does not improve after 2 epoch breat a go to next trial.
# save the model checkpoints, load and perform a test on test data, return metric such as acc, F1, recall, precision, PAUROC, and MCC

# return the full code after correction

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
import dgl
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from itertools import product

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from processing.dataprocessing import dataset


class GraphFunctionDataset(Dataset):
    def __init__(self, df, graph_dir, split='train', verbose=True):
        self.df = df[df['label'] == split]
        vuldf = self.df[self.df.vul == 1]
        nonvuldf = self.df[self.df.vul == 0]  #.sample(len(vuldf), random_state=0)  # to balanced the sample
        self.df = pd.concat([vuldf, nonvuldf])
        self.graph_dir = graph_dir
        self.graph_ids = []

        for graph_id in tqdm(self.df['id'].tolist(), desc=f"Checking graphs for {split}"):
            graph_path = os.path.join(self.graph_dir, f"{graph_id}")
            if os.path.exists(graph_path):
                self.graph_ids.append(graph_id)

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx):
        graph_id = self.graph_ids[idx]
        graph_path = os.path.join(self.graph_dir, f"{graph_id}")
        g = dgl.load_graphs(graph_path)[0][0]

        vul_label = self.df[self.df['id'] == graph_id]['vul'].values[0] # print this value to make sure
        g.ndata['_FVULN'] = torch.tensor([vul_label] * g.num_nodes())
        return g


class MultiTaskGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, dropout):
        super().__init__()
        self.gat1 = dgl.nn.GATConv(in_feats, hidden_feats, num_heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1, feat_drop=dropout, attn_drop=dropout)

        # MLP with dropout for node classification
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 2)
        )

        # MLP with dropout for graph classification
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 2)
        )

    def forward(self, g):
        # h = torch.cat([g.ndata['_RANDFEAT'], g.ndata['_FUNC_EMB'], g.ndata['_CODEBERT']], dim=1) # make this one work. to do, try to use all and reshape at the size of h = g.ndata['_CODEBERT'].float()
        h = g.ndata['_CODEBERT'].float()
        h = self.gat1(g, h).flatten(1)
        h = self.gat2(g, h).squeeze(1)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        node_logits = self.node_mlp(h)
        graph_logits = self.graph_mlp(hg)
        return node_logits, graph_logits


class LitMultiTaskGAT(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskGAT(config['in_feats'], config['hidden_feats'], config['num_heads'], config['dropout'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = config['lr']

    def forward(self, g):
        return self.model(g)

    def training_step(self, batch, batch_idx):
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()

        node_loss = self.loss_fn(node_logits, node_labels)
        func_loss = self.loss_fn(graph_logits.view(1, -1), func_label.view(1))
        loss = node_loss + func_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, graph_logits = self(batch)
        func_label = batch.ndata['_FVULN'][0].long()
        graph_pred = torch.argmax(graph_logits).item()
        acc = (graph_pred == func_label.item())
        self.log('val_acc', acc, prog_bar=True, batch_size = 1)
        return acc

    def test_step(self, batch, batch_idx):
        _, graph_logits = self(batch)
        func_label = batch.ndata['_FVULN'][0].long()
        graph_pred = torch.argmax(graph_logits).item()
        acc = (graph_pred == func_label.item())
        self.log('test_acc', acc, prog_bar=True, batch_size = 1)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


num_cpus = torch.get_num_threads()
def train_with_param_trials(df, graph_dir, config_grid, max_epochs=5):
    train_set = GraphFunctionDataset(df, graph_dir, split='train')
    val_set = GraphFunctionDataset(df, graph_dir, split='val')
    test_set = GraphFunctionDataset(df, graph_dir, split='test')

    train_loader = GraphDataLoader(train_set, batch_size=1, shuffle=True, num_workers=num_cpus) # add the num_workers=num_cpus
    val_loader = GraphDataLoader(val_set, batch_size=1, num_workers=num_cpus) # add the num_workers=num_cpus
    test_loader = GraphDataLoader(test_set, batch_size=1, num_workers=num_cpus) # add the num_workers=num_cpus

    best_val_acc = 0
    best_model_path = None
    best_config = None

    trials = list(product(config_grid['hidden_feats'], config_grid['dropout'], config_grid['lr']))
    print(f"Running {len(trials)} trials...")

    for idx, (hidden, dropout, lr) in enumerate(trials):
        print(f"\n=== Trial {idx+1} ===")
        trial_config = {
            'in_feats': config_grid['in_feats'],
            'hidden_feats': hidden,
            'num_heads': config_grid['num_heads'],
            'dropout': dropout,
            'lr': lr
        }

        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            dirpath=f"{utls.cache_dir()}/checkpoints",
            filename=f"trial-{idx+1}-{{val_acc:.4f}}"
        )

        model = LitMultiTaskGAT(trial_config)
        trainer = Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback], logger=False)

        trainer.fit(model, train_loader, val_loader)

        best_checkpoint_path = checkpoint_callback.best_model_path
        val_acc = checkpoint_callback.best_model_score.item()
        print(f"Trial {idx+1} val_acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = best_checkpoint_path
            best_config = trial_config

    print(f"\nBest model: {best_model_path} with val_acc = {best_val_acc:.4f}")
    print(f"Best config: {best_config}")

    best_model = LitMultiTaskGAT(best_config).load_from_checkpoint(best_model_path, config=best_config)
    trainer = Trainer(logger=False)
    trainer.test(best_model, test_loader)

    return best_model, best_config


if __name__ == '__main__':
    df = dataset()
    graph_dir = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"

    config_grid = {
        'in_feats': 768, # must e the same as the feature in graph
        'hidden_feats': [128, 256],
        'num_heads': 4,
        'dropout': [0.2, 0.3],
        'lr': [1e-4, 1e-3]
    }

    model, best_config = train_with_param_trials(df, graph_dir, config_grid, max_epochs=3)


# ask to stop id acc does not increase, use all feature and reshape them to code bert size
# use F-1 as target metric for finetune
# print our all the metric for the best trials

# correct from here. 
# Best model: /home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/checkpoints/trial-1-val_acc=0.9007.ckpt with val_acc = 0.9007
# Best config: {'in_feats': 768, 'hidden_feats': 128, 'num_heads': 4, 'dropout': 0.2, 'lr': 0.0001}
# Traceback (most recent call last):
#   File "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/./model/modelcscs.py", line 193, in <module>
#     }
#   File "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/./model/modelcscs.py", line 174, in train_with_param_trials
#     print(f"Best config: {best_config}")
#   File "/home/rz.lekeufack/Rosmael/.venv/lib/python3.10/site-packages/pytorch_lightning/utilities/model_helpers.py", line 121, in wrapper
#     raise TypeError(
# TypeError: The classmethod `LitMultiTaskGAT.load_from_checkpoint` cannot be called on an instance. Please call it on the class type and make sure the return value is used.