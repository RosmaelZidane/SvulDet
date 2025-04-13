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


import pandas as pd

# Sample DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)


print(df)

# Reverse the DataFrame using iloc
df = df.iloc[::-1]#.reset_index(drop=True)

print(df)