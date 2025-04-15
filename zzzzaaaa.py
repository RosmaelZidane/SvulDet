import json
import os
from tqdm import tqdm

idss = df.id.tolist()
ids = []

def check_validity(_id):
    """Check whether sample with id=_id has enough node/edges."""
    try:
        with open(f"{utls.processed_dir()}/dataset/before/{_id}.java.nodes.json", "r") as f:
            nodes = json.load(f)
            lineNums = set()
            for n in nodes:
                if "lineNumber" in n:
                    lineNums.add(n["lineNumber"])
                    if len(lineNums) > 1:
                        break
            if len(lineNums) <= 1:
                return False

        with open(f"{utls.processed_dir()}/dataset/before/{_id}.java.edges.json", "r") as f:
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
