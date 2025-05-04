# import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import uutils.__utils__ as utls

# print(utls.project_dir())

# print("-------")
# print(utls.storage_dir())
# print(f"+++++++++++++ \n{utls.external_dir()}")

# print("----------------------------------")
# print(dir(utls))


# import pandas as pd

# # Sample DataFrame
# data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
# df = pd.DataFrame(data)


# print(df)

# # Reverse the DataFrame using iloc
# df = df.iloc[::-1]#.reset_index(drop=True)

# print(df)



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


# sbert = SBERTEmbedder(f"{utls.cache_dir()}/embedmodel/SentenceBERT")
# word2vec = Word2VecEmbedder(f"{utls.cache_dir()}/embedmodel/Word2vec/Word2vec/word2vec_model.bin")

# # print(df.columns)
# text = df['before'][0]


# path = '/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/processed/dataset/before/1.c'

# with open(path, 'r') as f:
#     code = f.read()
    
# print(code)




# code = [code]
# # # print(word2vec)

# # # print(text)






# # # codebert
# codebert = CodeBertEmbedder(f"{utls.cache_dir()}/embedmodel/CodeBERT")

# featurescode = [codebert.embed([c]).detach().cpu() for c in code]
# print(f"codebert \n{featurescode}")






# featuressbert = [th.tensor(sbert.embed([c])).detach().cpu() for c in code]


# print(f"Sbert \n {featuressbert}")


# featuress = [th.tensor(word2vec.embed(c)).detach().cpu() for c in code]

# print(f"Word2vec \n{featuress}")

# # # print(sbert.embed(text))

# # # print(word2vec.embed(text))



# check validity


import json
import os
from tqdm import tqdm

idss = df.id.tolist()
ids = []

def check_validity(_id):
    """Check whether sample with id=_id has enough node/edges."""
    try:
        with open(f"{utls.processed_dir()}/dataset/before/{_id}.c.nodes.json", "r") as f:
            nodes = json.load(f)
            lineNums = set()
            for n in nodes:
                if "lineNumber" in n:
                    lineNums.add(n["lineNumber"])
                    if len(lineNums) > 1:
                        break
            if len(lineNums) <= 1:
                return False

        with open(f"{utls.processed_dir()}/dataset/before/{_id}.c.edges.json", "r") as f:
            edges = json.load(f)
            edge_set = set([i[2] for i in edges])
            if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                return False
        return True

    except Exception as E:
        print(f"[ERROR] {E} -- Skipped ID: {_id}")
        return False

# Apply validity check
for _id in tqdm(idss, desc="Validating samples"):
    if check_validity(_id):
        ids.append(_id)

print(f"Valid IDs found: {len(ids)}")



