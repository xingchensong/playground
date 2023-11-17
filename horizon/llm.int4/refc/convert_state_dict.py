import torch
import sys
import numpy as np

if len(sys.argv) != 3:
    print("Usage: convert-state_dict.py ./assets/mnist_model.state_dict " +
          "./assets/xcml-model-f32.txt\n")
    sys.exit(1)

state_dict_file = sys.argv[1]
fname_out = sys.argv[2]

state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))

list_vars = state_dict

fout = open(fname_out, "w")

"""
>>> a = torch.nn.Linear(in_features=3, out_features=2)
>>> a.weight.shape
torch.Size([2, 3])

可见weight的shape是out_features, in_features，即行数是输出维度，列数是输入维度

>>> a = torch.randn((2, 3))
>>> a
tensor([[-0.2258, -1.4521, -1.7965],
        [-0.2954, -1.5656,  1.7998]])
>>> a.reshape(-1)
tensor([-0.2258, -1.4521, -1.7965, -0.2954, -1.5656,  1.7998])

可见reshape后的数据是按行排列的,即先把输入维度的权重排完，再排输出维度的权重
"""
for name in list_vars.keys():
    print("Tensor {}, shape {}".format(name, list_vars[name].shape))
    data = list_vars[name].reshape(-1).numpy()
    data = data.astype(np.float32)
    for i in range(len(data)):
        fout.write("{}\n".format(data[i]))
fout.close()

print("Done. Output file: " + fname_out)
print("")
