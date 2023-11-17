ggml commit: aa1d26e

## MNIST with fully connected network

### Model definition

A fully connected layer + relu, followed by a **quantized** fully connected layer.

```py
input_size = 784   # img_size = (28,28) ---> 28*28=784 in total
hidden_size = 512  # number of nodes at hidden layer
num_classes = 10   # number of output classes discrete range [0,9]
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
`````````

### Computation graph:

1. fc1, all in fp32 mode
2. relu (unary), all in fp32 mode
3. fc2, q8_0 wight * q8_0 activation -> fp32 output, fp32 output + fp32 bias -> fp32 result

![mnist dot](https://user-images.githubusercontent.com/13466943/283752958-fa8de764-45a6-41ea-9e21-9731ef17f0ef.png)

### Running the example (ggml)

```bash
python train.py  # (optional, a pretrained model is provided in assests)
python convert-h5-to-ggml.py ./assets/mnist_model.state_dict ./assets/ggml-model-f32.bin
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 4
./build/bin/mnist-quantize ./assets/ggml-model-f32.bin ./assets/ggml-model-q8_0.bin q8_0
./build/bin/mnist ./assets/ggml-model-q8_0.bin ./assets/t10k-images.idx3-ubyte 6
```

### Running the example (xcsong-ml)

```bash
python convert_state_dict.py assets/mnist_model.state_dict assets/xcml-model-f32.txt  # (optional, a pre-converted model is provided in assests)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 4
./build/bin/mnist ./assets/xcml-model-f32.txt ./assets/t10k-images.idx3-ubyte 6
```
