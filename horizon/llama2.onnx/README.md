# 环境安装

```sh
conda create -n py3.10-torch2.x python=3.10
conda activate py3.10-torch2.x
pip install -r requirements.txt  -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
```


# 注意

1. onnx导出不支持复数（complex64，旋转位置编码中用到了），为了能顺利导出，复数运算改成了float。虽然位置编码值不一致了，但是对于目标是速度/结构分析而言，无大碍
2. 只能导出1层block的模型，超过1层的模型大小会超过2GB，onnx对超过2GB的模型会自动切分多个小onnx，无法形成一个完整文件，反而不利于分析，因此1层block足矣

# 使用

1. prefill 阶段模型导出 (以1层block的模型为例,2层超过2GB了，没法存成一个完整的onnx，onnx会自动分解成多个小onnx)

```sh
python export_onnx.py --n_layers 1 --input_length 1024 --cache_size 4096
```

2. decode 阶段模型导出 (以1层block的模型为例)

```sh
python export_onnx.py --n_layers 1 --input_length 1 --cache_size 4096
```

3. decode 阶段模型导出 (以1层block的模型为例,同时使用 groupconv 替代 nn.Linear)
```sh
python export_onnx.py --n_layers 1 --input_length 1 --cache_size 4096 --group_conv --block_size 32
```


文件目录如下：

```sh
./
|-- README.md
|-- exp
|   |-- llama2-7B-chat-hf-decode-inputlen1-nlayers1-cachesize4096-useGconv-block32.onnx
|   |-- llama2-7B-chat-hf-decode-inputlen1-nlayers1-cachesize4096.onnx
|   `-- llama2-7B-chat-hf-prefill-inputlen1024-nlayers1-cachesize4096.onnx
|-- export_onnx.py
|-- model
|   |-- __init__.py
|   |-- __pycache__
|   |-- group_conv.py
|   `-- model.py
|-- requirements.txt
```
