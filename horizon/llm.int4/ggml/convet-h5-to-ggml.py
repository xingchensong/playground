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
    print("Usage: convert-h5-to-ggml.py ./assets/mnist_model.state_dict " +
          "./assets/ggml-model-f16.bin\n")
    sys.exit(1)

state_dict_file = sys.argv[1]
fname_out = sys.argv[2]

state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))

list_vars = state_dict

fout = open(fname_out, "wb")
fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
fout.write(struct.pack("i", 784))  # n_input
fout.write(struct.pack("i", 512))  # n_hidden
fout.write(struct.pack("i", 10))   # n_classes
fout.write(struct.pack("i", 1))    # ftype, 1 for fp16, 0 for fp32

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    n_dims = len(data.shape)
    if name[-7:] == ".weight" and n_dims == 2:
        print("Tensor {}, shape {}, convert to fp16".format(name, data.shape))
        data = data.astype(np.float16)
        ftype = 1
    else:
        print("Tensor {}, shape {}, convert to fp32".format(name, data.shape))
        data = data.astype(np.float32)
        ftype = 0
    name = name.replace('.', '_')
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
