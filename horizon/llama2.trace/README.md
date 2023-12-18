# 环境安装

```sh
conda create -n trace python=3.8  # hbdk4只支持python3.8
conda activate trace
pip install -r requirements.txt  -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip install -U hbdk4-compiler -i https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
```


# 使用

1. prefill 阶段模型导出 (以2层block的模型为例)

```sh
python trace.py --n_layers 2 --input_length 1000 --cache_size 4096
```

2. decode 阶段模型导出 (以2层block的模型为例)

```sh
python trace.py --n_layers 2 --input_length 1 --cache_size 4096
```


文件目录如下：

```sh
.
|-- README.md
|-- exp
|   |-- llama2-7B-chat-hf-decode-inputlen1-nlayers2-cachesize4096.traced.pt
|   ·-- llama2-7B-chat-hf-prefill-inputlen1000-nlayers2-cachesize4096.traced.pt
|-- model
|   |-- model.py
|   ·-- model_per_block.py
|-- requirements.txt
·-- trace.py
```
