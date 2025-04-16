import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
import dgl
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from processing.dataprocessing import dataset

class GraphFunctionDataset(Dataset):
    def __init__(self, df, graph_dir, split='train', verbose=True):
        self.df = df[df['label'] == split]
        vuldf = self.df[self.df.vul == 1]
        nonvuldf = self.df[self.df.vul == 0].sample(len(vuldf), random_state=0)
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
        vul_label = self.df[self.df['id'] == graph_id]['vul'].values[0]
        g.ndata['_FVULN'] = torch.tensor([vul_label] * g.num_nodes())
        return g

class MultiTaskGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, dropout):
        super().__init__()
        self.gat1 = dgl.nn.GATConv(in_feats, hidden_feats, num_heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1, feat_drop=dropout, attn_drop=dropout)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 2)
        )

        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 2)
        )

    def forward(self, g):
        rand_feat = g.ndata['_RANDFEAT'].float()
        func_emb = g.ndata['_FUNC_EMB'].float()
        codebert = g.ndata['_CODEBERT'].float()

        func_emb = nn.functional.interpolate(func_emb.unsqueeze(0), size=codebert.shape[1], mode='nearest').squeeze(0)
        rand_feat = nn.functional.interpolate(rand_feat.unsqueeze(0), size=codebert.shape[1], mode='nearest').squeeze(0)

        h = torch.cat([rand_feat, func_emb, codebert], dim=1)
        h = nn.Linear(h.shape[1], codebert.shape[1]).to(h.device)(h)

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
        self.val_preds = []
        self.val_labels = []

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
        pred = torch.argmax(graph_logits).item()
        self.val_preds.append(pred)
        self.val_labels.append(func_label.item())

    def on_validation_epoch_end(self):
        f1 = f1_score(self.val_labels, self.val_preds, zero_division=0)
        self.log('val_f1', f1, prog_bar=True)
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        _, graph_logits = self(batch)
        func_label = batch.ndata['_FVULN'][0].long()
        pred = torch.argmax(graph_logits).item()
        return {'pred': pred, 'true': func_label.item()}

    def test_epoch_end(self, outputs):
        preds = [x['pred'] for x in outputs]
        labels = [x['true'] for x in outputs]
        metrics = {
            'acc': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, zero_division=0),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'auroc': roc_auc_score(labels, preds),
            'mcc': matthews_corrcoef(labels, preds)
        }
        for k, v in metrics.items():
            self.log(f'test_{k}', v, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train_with_param_trials(df, graph_dir, config_grid, max_epochs=5):
    num_cpus = torch.get_num_threads()
    train_set = GraphFunctionDataset(df, graph_dir, split='train')
    val_set = GraphFunctionDataset(df, graph_dir, split='val')
    test_set = GraphFunctionDataset(df, graph_dir, split='test')

    train_loader = GraphDataLoader(train_set, batch_size=1, shuffle=True, num_workers=num_cpus)
    val_loader = GraphDataLoader(val_set, batch_size=1, num_workers=num_cpus)
    test_loader = GraphDataLoader(test_set, batch_size=1, num_workers=num_cpus)

    best_val_f1 = 0
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
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            dirpath=f"{utls.cache_dir()}/checkpoints",
            filename=f"trial-{idx+1}-{{val_f1:.4f}}"
        )

        early_stop_callback = EarlyStopping(monitor='val_f1', patience=2, mode='max')

        model = LitMultiTaskGAT(trial_config)
        trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=False,
            accelerator="auto"
        )

        trainer.fit(model, train_loader, val_loader)

        best_checkpoint_path = checkpoint_callback.best_model_path
        val_f1 = checkpoint_callback.best_model_score.item()
        print(f"Trial {idx+1} val_f1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = best_checkpoint_path
            best_config = trial_config

    print(f"\nBest model: {best_model_path} with val_f1 = {best_val_f1:.4f}")
    print(f"Best config: {best_config}")

    best_model = LitMultiTaskGAT.load_from_checkpoint(best_model_path, config=best_config)
    trainer = Trainer(logger=False)
    trainer.test(best_model, test_loader)

    return best_model, best_config

if __name__ == '__main__':
    seed_everything(42)
    df = dataset()
    graph_dir = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"

    config_grid = {
        'in_feats': 768,
        'hidden_feats': [128],
        'num_heads': 4,
        'dropout': [0.2],
        'lr': [1e-4, 1e-3]
    }

    model, best_config = train_with_param_trials(df, graph_dir, config_grid, max_epochs=5)



# code to save the history of the f1 score and plot.
# code to evaluate function level and node level and save metrics