# 环境安装

```sh
conda create -n trace python=3.8  # hbdk4只支持python3.8
conda activate trace
pip install -r requirements.txt  -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
pip install -U hbdk4-compiler -i https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
```
