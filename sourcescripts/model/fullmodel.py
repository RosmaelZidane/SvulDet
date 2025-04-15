import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
import dgl
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
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
        self.verbose = verbose

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
        return g


class MultiTaskGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, dropout=0.3):
        super().__init__()
        self.gat1 = dgl.nn.GATConv(in_feats, hidden_feats, num_heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1, feat_drop=dropout, attn_drop=dropout)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats // 2, 2)
        )
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats // 2, 2)
        )

    def forward(self, g):
        h = torch.cat([g.ndata['_RANDFEAT'], g.ndata['_FUNC_EMB'], g.ndata['_CODEBERT']], dim=1)
        h = self.gat1(g, h)
        h = F.elu(h.flatten(1))
        h = self.gat2(g, h).squeeze(1)

        node_logits = self.node_classifier(h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        graph_logits = self.graph_classifier(hg)

        return node_logits, graph_logits


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
        'recall': recall_score(y_true, y_pred, average="macro", zero_division=0),
        'f1': f1_score(y_true, y_pred, average="macro", zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }


def evaluate(model, dataloader, device):
    model.eval()
    node_y_true, node_y_pred, node_y_prob = [], [], []
    func_y_true, func_y_pred, func_y_prob = [], [], []

    with torch.no_grad():
        for g in tqdm(dataloader, desc="Evaluating"):
            g = g.to(device)
            node_logits, graph_logits = model(g)
            node_labels = g.ndata['_VULN'].long()
            func_label = g.ndata['_FVULN'].long()[0]

            node_probs = F.softmax(node_logits, dim=1)[:, 1]
            node_preds = torch.argmax(node_logits, dim=1)

            graph_probs = F.softmax(graph_logits, dim=1)[1].item()
            graph_pred = torch.argmax(graph_logits).item()

            node_y_true.extend(node_labels.cpu().numpy())
            node_y_pred.extend(node_preds.cpu().numpy())
            node_y_prob.extend(node_probs.cpu().numpy())

            func_y_true.append(func_label.item())
            func_y_pred.append(graph_pred)
            func_y_prob.append(graph_probs)

    return {
        'node': compute_metrics(node_y_true, node_y_pred, node_y_prob),
        'function': compute_metrics(func_y_true, func_y_pred, func_y_prob)
    }


def train_model(df, graph_dir, config, save_plot_path='training_plot.pdf'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = GraphFunctionDataset(df, graph_dir, split='train')
    val_set = GraphFunctionDataset(df, graph_dir, split='val')
    test_set = GraphFunctionDataset(df, graph_dir, split='test')

    train_loader = GraphDataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = GraphDataLoader(val_set, batch_size=1)
    test_loader = GraphDataLoader(test_set, batch_size=1)

    model = MultiTaskGAT(config['in_feats'], config['hidden_feats'], config['num_heads'], config['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for g in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            g = g.to(device)
            node_logits, graph_logits = model(g)

            node_labels = g.ndata['_VULN'].long()
            func_label = g.ndata['_FVULN'].long()[0].unsqueeze(0)

            node_loss = loss_fn(node_logits, node_labels)
            func_loss = loss_fn(graph_logits.unsqueeze(0), func_label)
            loss = node_loss + func_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_metrics['function']['accuracy'])

        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Func Acc = {val_metrics['function']['accuracy']:.4f}")

    test_metrics = evaluate(model, test_loader, device)
    print("\nTest Metrics:")
    print("Node-level:", test_metrics['node'])
    print("Function-level:", test_metrics['function'])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Func Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_plot_path)
    print(f"Plot saved to {save_plot_path}")

    return model, history


# === USAGE ===
if __name__ == '__main__':
    df = dataset()
    
    dfvul = df[df['vul'] == 1]
    dfnvul = df[df['vul'] == 0].sample(len(dfvul))
    df = pd.concat([dfvul, dfnvul])
    graph_dir = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
    config = {
        'in_feats': 1636,
        'hidden_feats': 128,
        'num_heads': 4,
        'dropout': 0.3,
        'lr': 1e-4,
        'epochs': 30
    }
    model, history = train_model(df, graph_dir, config)

    # To tune, wrap train_model with Ray's tune.with_parameters
    # and configure the search space and scheduler.
