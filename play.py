import pickle
import numpy as np
import os

path = "/Users/sudgarg/work/coding/nanoGPT/data/shakespeare_char/meta.pkl"

# Load object back
with open(path, "rb") as f:
    loaded_data = pickle.load(f)
    print(loaded_data)

data_dir = "/Users/sudgarg/work/coding/nanoGPT/data/shakespeare_char"
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
print(data)