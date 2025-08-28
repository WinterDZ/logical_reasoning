#!/usr/bin/env python
# coding: utf-8

# ## 配置环境

# In[1]:


import json
import os
from pprint import pprint
import re
from tqdm import tqdm
import random

import uuid
import openai
from openai import OpenAI
import tiktoken
import json
import numpy as np
import requests
from scipy import sparse
#from rank_bm25 import BM25Okapi
#import jieba
from http import HTTPStatus


from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import json
import time
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
import datetime

logger.remove()  # 移除默认的控制台输出
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip")

MODEL_NAME = 'Qwen2-7B-Instruct-lora'


# ## 下载大模型

# In[2]:





# ## 模型设置

# In[22]:


## 测试文件及输出文件名
filename = "./submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'+'.jsonl')
filetest = "./data/round1_test_data.jsonl"


# In[7]:


device = "cuda"
model_path = './model/merged_model_an'
llm = LLM(model_path,gpu_memory_utilization=0.9, max_model_len=5000) # 使用vllm.LLM()创建LLM对象
tokenizer = AutoTokenizer.from_pretrained(model_path) # 使用AutoTokenizer.from_pretrained()创建tokenizer


# In[9]:


# 这里定义了prompt推理模版

def get_prompt(problem, question, options):

    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
"""
    # print(prompt)
    return prompt




# 这里使用extract抽取模获得抽取的结果

def extract(input_text):
    # ans_pattern = re.compile(r"(.)", re.S)

    ans_pattern = re.compile(r"答案是：(.)", re.S)
    problems = ans_pattern.findall(input_text)
    # print(problems)
    if(problems == []):
        return 'A'
    return problems[0]


# In[11]:


def most_frequent_char(char1, char2, char3):
    # 创建一个字典来存储每个字符的出现次数
    frequency = {char1: 0, char2: 0, char3: 0}
    
    # 增加每个字符的出现次数
    frequency[char1] += 1
    frequency[char2] += 1
    frequency[char3] += 1
    
    # 找到出现次数最多的字符
    most_frequent = max(frequency, key=frequency.get)
    
    return most_frequent


# In[12]:


def process_datas(datas, MODEL_NAME):
    prompts = []
    results = []
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用第1块GPU
    
    # 获取每个问题的prompt，并将prompt信息装入messages，（关键）再应用tokenizer的对话模板
    for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
        problem = data['problem']
        for id, question in enumerate(data['questions']):
            prompt = get_prompt(
                problem, 
                question['question'], 
                question['options'],
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text) # 将处理完成的prompt添加入prompts列表，准备输入vllm批量推理
    
    # 定义推理参数
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    
    # 开始推理
    # 单路投票推理
    # outputs = llm.generate(prompts, sampling_params)
    # 多路投票推理（这里通过进行三次推理，模仿多路投票的过程）
    outputs1 = llm.generate(prompts, sampling_params)
    outputs2 = llm.generate(prompts, sampling_params)
    outputs3 = llm.generate(prompts, sampling_params)

    '''
    单路投票
    '''
    # i = 0
    # for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
    #     for id, question in enumerate(data['questions']):
    #         generated_text = outputs[i].outputs[0].text
    #         i = i + 1
    #         extract_response= extract(generated_text)
    #         data['questions'][id]['answer'] = extract_response
    #         results.append(data)

    '''
    多路投票
    '''
    i = 0 # 由于outputs中存储的回答序号并不是与datas中的序号一一对应（因为一个问题背景下可能有多个问题），因此使用一个计数变量另外遍历outputs
    for data in tqdm(datas, desc="Extracting answers", total=len(datas)):
        for id, question in enumerate(data['questions']):
            # 获取每一路推理的回答文本
            generated_text1 = outputs1[i].outputs[0].text
            generated_text2 = outputs2[i].outputs[0].text
            generated_text3 = outputs3[i].outputs[0].text
            i = i + 1
            # 从文本中提取答案选项
            extract_response1, extract_response2, extract_response3 = extract(generated_text1),  extract(generated_text2),  extract(generated_text3)
            # 投票选择出现次数最多的选项作为答案
            ans = most_frequent_char(extract_response1, extract_response2, extract_response3)
            data['questions'][id]['answer'] = ans
            results.append(data)

    return results


# In[13]:


def main(ifn, ofn):
    if os.path.exists(ofn):
        pass
    data = []
    # 按行读取数据
    with open(ifn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    datas = data
    # print(data)
    # 均匀地分成多个数据集
    return_list = process_datas(datas,MODEL_NAME)
    print(len(return_list))
    print("All tasks finished!")
    return return_list


# In[14]:


if __name__ == '__main__':
    return_list = main(filetest,filename)


# ## 结果优化

# In[15]:


def has_complete_answer(questions):
    # 这里假设完整答案的判断逻辑是：每个question都有一个'answer'键
    for question in questions:
        if 'answer' not in question:
            return False
    return True

def filter_problems(data):
    result = []
    problem_set = set()

    for item in data:
        # print('处理的item' ,item)
        problem = item['problem']
        if problem in problem_set:
            # 找到已存在的字典
            for existing_item in result:
                if existing_item['problem'] == problem:
                    # 如果当前字典有完整答案，替换已存在的字典
                    if has_complete_answer(item['questions']):
                        existing_item['questions'] = item['questions']
                        existing_item['id'] = item['id']
                    break
        else:
            # 如果当前字典有完整答案，添加到结果列表
            if has_complete_answer(item['questions']):
                result.append(item)
                problem_set.add(problem)

    return result


# In[16]:


return_list
return_list = filter_problems(return_list)
sorted_data = sorted(return_list, key=lambda x: int(str(x['id'])[-3:]))
print(sorted_data)


# In[17]:


return_list


# In[18]:


def find_missing_ids(dict_list):
    # 提取所有序号
    extracted_ids = {int(d['id'][-3:]) for d in dict_list}
    
    # 创建0-500的序号集合
    all_ids = set(range(500))
    
    # 找出缺失的序号
    missing_ids = all_ids - extracted_ids
    
    return sorted(missing_ids)

# 示例字典列表
dict_list = sorted_data

# 找出缺失的序号
missing_ids = find_missing_ids(dict_list)
print("缺失的序号:", missing_ids)


# In[19]:


len(missing_ids)


# In[20]:


data  = []
with open(filetest) as reader:
    for id,line in enumerate(reader):
        if(id in missing_ids):
            sample = json.loads(line)
            for question in sample['questions']:
                question['answer'] = 'A'
            sorted_data.append(sample)
sorted_data = sorted(sorted_data, key=lambda x: int(str(x['id'])[-3:]))
        


# ## 输出结果

# In[23]:


with open(filename, 'w') as writer:
    for sample in sorted_data:
        writer.write(json.dumps(sample, ensure_ascii=False))
        writer.write('\n')


# In[ ]:




