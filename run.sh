#!/bin/bash

# 安装必要的依赖库
pip install scipy openai tiktoken retry dashscope loguru
pip install -U ipywidgets
pip install modelscope==1.9.5
pip install "transformers>=4.39.0"
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install transformers_stream_generator==0.0.4
pip install datasets==2.18.0
pip install peft==0.10.0
pip install openai==1.17.1
pip install tqdm==4.64.1
pip install transformers==4.39.3
python -m pip install setuptools==69.5.1
pip install vllm==0.4.0.post1
pip install nest-asyncio
pip install accelerate
pip install tf-keras

# 运行 merge_model.py 文件的前半部分，只执行下载模型的代码
echo "下载模型中..."
python3 -c "from modelscope import snapshot_download; snapshot_download('qwen/qwen2-7b-instruct', cache_dir='../model/', revision='master')"

# 清理 GPU 内存
echo "清理 GPU 内存..."
python3 -c "import torch; torch.cuda.empty_cache()"

# 等待 5 秒确保操作完成
sleep 5

# 运行 merge_model.py 文件的剩余部分，继续合并模型
echo "继续执行模型合并..."
python3 src/merge_model.py

# 清理 GPU 内存
echo "清理 GPU 内存..."
python3 -c "import torch; torch.cuda.empty_cache()"

# 等待 5 秒确保操作完成
sleep 5

# 运行逻辑推理赛道文件
echo "运行逻辑推理赛道脚本..."
python3 src/逻辑推理赛道_学习小队_136_18917266950.py

# 输出提示信息
echo "程序运行完成"
