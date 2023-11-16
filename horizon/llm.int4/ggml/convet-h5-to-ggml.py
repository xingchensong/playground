# Convert MNIS h5 transformer model to ggml format
#
# Load the (state_dict) saved model using PyTorch
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# At the start of the ggml file we write the model parameters

import sys
import struct
import numpy as np


import torch

if len(sys.argv) != 3:
    print("Usage: convert-h5-to-ggml.py model model.out\n")
    sys.exit(1)

state_dict_file = sys.argv[1]
fname_out = sys.argv[2]

state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))

list_vars = state_dict
print(list_vars)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)
    n_dims = len(data.shape)
    fout.write(struct.pack("i", n_dims))
    data = data.astype(np.float32)
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
