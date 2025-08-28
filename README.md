## 环境依赖
### 镜像版本：
modelscope:1.14.0
pytorch:2.1.2
tensorflow:2.14.0
python:3.10
cuda:12.1
ubuntu22.04
基于魔搭modelscope:1.14.0-pytorch2.1.2tensorflow2.14.0-gpu-py310-cu121-ubuntu22.04镜像完成模型内容。

### 需要安装的库:
scipy 
openai 
tiktoken 
retry 
dashscope 
loguru 
-U ipywidgets 
modelscope==1.9.5 
"transformers>=4.39.0" 
streamlit==1.24.0 
sentencepiece==0.1.99
transformers_stream_generator==0.0.4
datasets==2.18.0
peft==0.10.0
openai==1.17.1
tqdm==4.64.1
transformers==4.39.3
setuptools==69.5.1
vllm==0.4.0.post1
nest-asyncio
accelerate
tf-keras


## 方法介绍
1.部署离线vllm，加快模型推理速度。
2.使用三路投票，提高答案准确率。
3.使用lora微调，采用baseline2中的ana作为微调训练文件。根据训练效果，采用第1800步的训练数据，即loss为0.05左右，此时微调效果最好。


## 代码结构
merge_model.py用来将微调的文件与qwen2-7b-instruct大模型进行融合，输出为Qwen2-7B-Instruct-lora模型并存储到output中。

逻辑推理赛道_学习小队_136_18917266950.py为推理模型，包含以下部分。
### 模型设置
部署离线vllm、设置提示词模板、抽取答案、三路投票、以及模型推理过程，运行以上代码，得到微调大模型的结果，存入return_list中。
### 结果优化
分析微调大模型输出结果，找出错漏确实的序号进行补充，再按序号进行排序。
### 输出结果
将结果输出至submit文件夹中，按时间进行命名。


## 文件结构
project
    |--data
        |--round1_test_data.jsonl
        |--round1_train_data.jsonl
        |--external_data
    |--model
        |--merged_model_an
        |--qwen
            |--qwen2-7b-instruct
    |--src
        |--output
            |--Qwen2_instruct_lora
                |--checkpoint-1800
                    |--adapter_config.json
                    |--adapter_model.safetensors
                    |--optimizer.pt
                    |--README.md
                    |--rng_state.pth
                    |--scheduler.pt
                    |--trainer_state.json
                    |--training_args.bin
        |--merge_model.py
        |--逻辑推理赛道_学习小队_136_18917266950.py
    |--submit
        |-- submit_XXXX.jsonl
    |--README.md
    |--run.sh


