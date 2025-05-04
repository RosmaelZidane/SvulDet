# import json
# import os
# from tqdm import tqdm
import random

# idss = df.id.tolist()
# ids = []

# def check_validity(_id):
#     """Check whether sample with id=_id has enough node/edges."""
#     try:
#         with open(f"{utls.processed_dir()}/dataset/before/{_id}.c.nodes.json", "r") as f:
#             nodes = json.load(f)
#             lineNums = set()
#             for n in nodes:
#                 if "lineNumber" in n:
#                     lineNums.add(n["lineNumber"])
#                     if len(lineNums) > 1:
#                         break
#             if len(lineNums) <= 1:
#                 return False

#         with open(f"{utls.processed_dir()}/dataset/before/{_id}.c.edges.json", "r") as f:
#             edges = json.load(f)
#             edge_set = set([i[2] for i in edges])
#             if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
#                 return False
#         return True

#     except Exception as E:
#         print(f"[ERROR] {E} -- Skipped ID: {_id}")
#         return False

# # Apply validity check
# for _id in tqdm(idss, desc="Validating samples"):
#     if check_validity(_id):
#         ids.append(_id)

# print(f"Valid IDs found: {len(ids)}")



import pandas as pd     

paths  = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/external/data_FFmpeg+qemu.csv"

df = pd.read_csv(paths)

df['id'] = df.index
# df['commit_ID'] = df['commit_id']
df['Domain_decsriptions'] = ""
df["Description_Mitre"] = ""
df['Sample Code'] = ""
df['diff_lines'] = ""


keepcols = ['id', 'commit_ID', 'CVE-ID', 'CWE-ID', 'project',
            'func_before','func_after', 'diff_lines', 'vul', 'Domain_decsriptions', 
            'Description_Mitre','Sample Code','diff_lines']
df = df[keepcols]



# Function to randomly add or remove a line of code
def modify_function(func_code):
    lines = func_code.split('\n')
    
    # Randomly decide to add or remove a line
    if random.choice([True, False]):  # Randomly choose to add or remove
        # Add a line of code
        new_line = '    return 0   // New line of code added'
        lines.insert(random.randint(1, len(lines)), new_line)  # Insert at a random position
    else:
        # Remove a random line if there are more than 1 line
        if len(lines) > 2:  # Keep the function structure intact
            lines.pop(random.randint(1, len(lines) - 1))  # Remove a random line

    return '\n'.join(lines)

# Modify functions where 'vul' == 1
df.loc[df['vul'] == 1, 'func_after'] = df.loc[df['vul'] == 1, 'func_after'].apply(modify_function)


df.to_csv(paths, index = False)








print(df.columns)





path2 = "/home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/external/ProjectKB_domain_csv.csv"
df2 = pd.read_csv(path2)


print(df2.columns)



print(df.id.tolist)