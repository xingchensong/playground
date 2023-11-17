ggml commit: aa1d26e

## MNIST with fully connected network

A fully connected layer + relu, followed by a **quantized** fully connected layer.

Computation graph:

![mnist dot](https://user-images.githubusercontent.com/13466943/283702577-ed2d82b6-2728-4500-9f44-6223c6822dc8.png)


### Running the example

```bash
python train.py  # (optional, a pretrained model is provided in assests)
python convert-h5-to-ggml.py ./assets/mnist_model.state_dict ./assets/ggml-model-f16.bin
./build/bin/mnist-quantize ./assets/ggml-model-f16.bin ./assets/ggml-model-q8_0.bin q8_0
./build/bin/mnist ./assets/ggml-model-q8_0.bin ./assets/t10k-images.idx3-ubyte
```
