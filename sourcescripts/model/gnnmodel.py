

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.dataprocessing import dataset
import uutils.__utils__ as utls

from tqdm import tqdm
import pandas as pd           



# Dataset class
class GraphFunctionDataset(Dataset):
    def __init__(self, df, graph_dir, split='train', verbose=True):
        self.df = df[df['label'] == split]
        # balance the function level
        vuldf = self.df[self.df.vul == 1]
        print(f'-------------------------------{len(vuldf)}')
        nonvuldf = self.df[self.df.vul == 0].sample(len(vuldf), random_state=0)
        self.df = pd.concat([vuldf, nonvuldf])
        self.graph_dir = graph_dir
        self.graph_ids = []
        self.verbose = verbose

        if self.verbose:
            print(f"Loading graph paths for split: {split}")
        for graph_id in tqdm(self.df['id'].tolist(), desc=f"Checking graphs for {split}"):
            graph_path = os.path.join(self.graph_dir, f"{graph_id}")
            if os.path.exists(graph_path):
                self.graph_ids.append(graph_id)
            elif self.verbose:
                pass
                # print(f"[Warning] Graph file not found: {graph_path}")

        if self.verbose:
            print(f"Total graphs found for {split}: {len(self.graph_ids)}")

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx):
        graph_id = self.graph_ids[idx]
        graph_path = os.path.join(self.graph_dir, f"{graph_id}")
        g = dgl.load_graphs(graph_path)[0][0]
        return g

# GAT Model
class GATClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, num_classes, dropout=0.3):
        super().__init__()
        self.gat1 = dgl.nn.GATConv(in_feats, hidden_feats, num_heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1, feat_drop=dropout, attn_drop=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats // 2, num_classes)
        )

    def forward(self, g):
        h = torch.cat([
            g.ndata['_RANDFEAT'],
            g.ndata['_FUNC_EMB'],
            g.ndata['_CODEBERT']
        ], dim=1)
        h = self.gat1(g, h)
        h = F.elu(h.flatten(1))
        h = self.gat2(g, h)
        h = h.squeeze(1)
        out = self.classifier(h)
        return out

# Training and Evaluation Functions
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for g in tqdm(dataloader, desc="Evaluating"):
            g = g.to(device)
            logits = model(g)
            labels = g.ndata['_VULN'].long().to(device)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred,average="macro", zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0,average="macro"),
        'roc_auc': roc_auc_score(y_true, y_prob,  average="macro"),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    stat = pd.DataFrame({"True labal": y_true,
                         "Predict": y_pred})
    stat.to_csv(f"{utls.outputs_dir()}/Predict_stat.csv", index=False)
    return metrics

def train_model(df, graph_dir, in_feats=1636, hidden_feats=128, 
                num_heads=4, num_classes=2, epochs=50, lr=1e-5, 
                batch_size=1, verbose=True, save_plot_path='training_plot.pdf'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = GraphFunctionDataset(df, graph_dir, split='train', verbose=verbose)
    val_set = GraphFunctionDataset(df, graph_dir, split='val', verbose=verbose)
    test_set = GraphFunctionDataset(df, graph_dir, split='test', verbose=verbose)

    
    train_loader = GraphDataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_set, batch_size=batch_size)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size)

    model = GATClassifier(in_feats, hidden_feats, num_heads, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs}")
        for g in tqdm(train_loader, desc="Training"):
            g = g.to(device)
            logits = model(g)
            labels = g.ndata['_VULN'].long().to(device)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        if verbose:
            print(f"Loss: {avg_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")

    # Final test evaluation
    test_metrics = evaluate(model, test_loader, device)
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    # Plot training loss and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_plot_path)
    print(f"\nTraining plot saved to {save_plot_path}")

    return model, history

# === USAGE INSTRUCTION ===



df = dataset()
# print(df.columns)

# storage/cache/Graph/dataset_svuldet_codebert_pdg+raw
graph_dir = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
model, history = train_model(df, graph_dir)
