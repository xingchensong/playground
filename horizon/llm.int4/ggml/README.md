ggml commit: aa1d26e

## MNIST with fully connected network

A fully connected layer + relu, followed by a fully connected layer + softmax.

Computation graph:

![mnist dot](https://user-images.githubusercontent.com/1991296/231882071-84e29d53-b226-4d73-bdc2-5bd6dcb7efd1.png)


### Running the example

```bash
python convert-h5-to-ggml.py ./assets/mnist_model.state_dict ./assets/ggml-model-f32.bin
./build/bin/mnist ./assets/ggml-model-f32.bin ./assets/t10k-images.idx3-ubyte
```
