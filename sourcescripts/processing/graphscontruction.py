"""The graph construction process is expected to be completed three times using different methods. 
This involves applying three embedding techniques: CodeBERT, Word2Vec, and SentenceBER"""


import os
import sys
from dgl import load_graphs, save_graphs
import dgl
import torch as th
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from embeddmodel.codebert import CodeBertEmbedder 
from embeddmodel.sentencebert import SBERTEmbedder
from embeddmodel.word2vec import Word2VecEmbedder
from dataprocessing import dataset, feature_extraction
from dataprocessing import get_dep_add_lines_dataset

df = dataset()

def initialize_lines_and_features(gtype="pdg", feat="all"):
    lines = get_dep_add_lines_dataset()
    lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
    return lines, gtype, feat

def cache_codebert_method_level(df, codebert, _id):
    savedir = utls.get_dir(utls.cache_dir() / "Graph/codebert_method_level") 
    batch_texts = df.before.tolist()
    texts = ["</s> " + ct for ct in batch_texts]
    embedded = codebert.embed(texts).detach().cpu()
    th.save(embedded, savedir / f"{_id}.pt")
    
def cache_sbert_method_level(df, sbert, _id):
    savedir = utls.get_dir(utls.cache_dir() / "Graph/sbert_method_level") 
    batch_texts = df.before.tolist()
    embedded = sbert.embed(batch_texts)
    th.save(embedded, savedir / f"{_id}.pt")
    
def cache_worde2vec_method_level(df, word2vec, _id):
    savedir = utls.get_dir(utls.cache_dir() / "Graph/word2vec_method_level") 
    text = df.before.tolist()[0]
    embedded = word2vec.embed(text)
    th.save(embedded, savedir / f"{_id}.pt")

def process_item(_id, df, codebert=None, word2vec=None, sbert=None, lines=None, graph_type="pdg", feat="all"):
    if codebert:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_codebert_{graph_type}") / str(_id)
    elif word2vec:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_word2vec_{graph_type}") / str(_id)
    elif sbert:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_sbert_{graph_type}") / str(_id)
    else:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_randfeat_{graph_type}") / str(_id)

    if os.path.exists(savedir):
        g = load_graphs(str(savedir))[0][0]
        return g

    code, lineno, ei, eo, et = feature_extraction(
        f"{utls.processed_dir()}/dataset/before/{_id}.java", graph_type)
    vuln = [1 if i in lines[_id] else 0 for i in lineno] if _id in lines else [0 for _ in lineno]

    g = dgl.graph((eo, ei))
    code = [c.replace("\\t", "").replace("\\n", "") for c in code]

    if codebert:
        features = [codebert.embed([c]).detach().cpu() for c in code]
        g.ndata["_CODEBERT"] = th.cat(features)

    if word2vec:
        features = [word2vec.embed(c) for c in code]
        g.ndata["_WORD2VEC"] = th.tensor(features).float()

    if sbert:
        features = [sbert.embed([c]) for c in code]
        g.ndata["_SBERT"] = th.tensor(features).float()

    g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
    g.ndata["_LINE"] = th.tensor(lineno).int()
    g.ndata["_VULN"] = th.tensor(vuln).float()
    g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes(),))
    g.edata["_ETYPE"] = th.tensor(et).long()

    if codebert:
        emb_path = utls.cache_dir() / f"Graph/codebert_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
    if sbert:
        emb_path = utls.cache_dir() / f"Graph/sbert_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
    if word2vec:
        emb_path = utls.cache_dir() / f"Graph/word2vec_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))

    g = dgl.add_self_loop(g)
    save_graphs(str(savedir), [g])
    return g

def cache_all_items(df, lines, graph_type="pdg", feat="all"):
    embedders = {
        "codebert": CodeBertEmbedder(f"{utls.cache_dir()}/embedmodel/CodeBERT"),
        "word2vec": Word2VecEmbedder(f"{utls.cache_dir()}/embedmodel/Word2Vec/word2vec_model.bin"),
        "sbert": SBERTEmbedder(f"{utls.cache_dir()}/embedmodel/SentenceBERT")
    }

    for name, embedder in embedders.items():
        print(f"\n--- Processing with {name.upper()} ---")
        for _id in tqdm(df.sample(len(df)).id.tolist()):
            try:
                process_item(
                    _id, df,
                    codebert=embedder if name == "codebert" else None,
                    word2vec=embedder if name == "word2vec" else None,
                    sbert=embedder if name == "sbert" else None,
                    lines=lines,
                    graph_type=graph_type,
                    feat=feat
                )
            except Exception as e:
                print(f"Error processing {_id} with {name}: {e}")

if __name__ == "__main__":
    lines, graph_type, feat = initialize_lines_and_features(gtype="pdg+raw", feat="all")

    codebert = CodeBertEmbedder(f"{utls.cache_dir()}/embedmodel/CodeBERT")
    sbert = SBERTEmbedder(f"{utls.cache_dir()}/embedmodel/SentenceBERT")
    word2vec = Word2VecEmbedder(f"{utls.cache_dir()}/embedmodel/Word2Vec/word2vec_model.bin")

    _ids = df.id.tolist()
    for _id in _ids:
        try:
            if not os.path.exists(utls.cache_dir() / f"Graph/codebert_method_level/{_id}.pt"):
                cache_codebert_method_level(df[df.id == _id], codebert, _id)
            if not os.path.exists(utls.cache_dir() / f"Graph/sbert_method_level/{_id}.pt"):
                cache_sbert_method_level(df[df.id == _id], sbert, _id)
            if not os.path.exists(utls.cache_dir() / f"Graph/word2vec_method_level/{_id}.pt"):
                cache_worde2vec_method_level(df[df.id == _id], word2vec, _id)
        except Exception as e:
            print(f"Error creating method-level embedding for {_id}: {e}")

    cache_all_items(df, lines, graph_type, feat)
