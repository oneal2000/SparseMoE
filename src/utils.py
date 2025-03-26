import os
import re
import json
import torch
import string
import numpy as np
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

from root_dir_path import ROOT_DIR
from prompt_template import get_prompt

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "3.27")

class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text) # 去除英文冠词
        def white_space_fix(text):
            return ' '.join(text.split())  # 去除多余空格
        def remove_punc(text):
            exclude = set(string.punctuation)  # 获取所有标点符号
            return ''.join(ch for ch in text if ch not in exclude)  # 去除所有标点
        def lower(text):
            return text.lower()  # 转换为小写
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score( # 计算预测答案是否与标准答案完全匹配。
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score( # 计算F1 分数，即预测和标准答案在 token 级别的重叠程度
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


def load_data(data_name, data_type, model_name):
    solve_dataset = []
    input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
    files = [f for f in os.listdir(input_dir)]


    if len(files) > 1: # 如果该目录下的 JSON 文件数大于 1，说明数据集中包含多个不同类型的子集
        if data_type == "total": # 合并所有文件的数据，是从filename到file内容的映射
            all_data = {}
            for filename in files:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    all_data[filename] = json.load(fin)
                    
            total_data = []
            idx = {filename: 0 for filename in files} # 记录每个 JSON 文件当前读取的索引
            for data in all_data["total.json"]: # 遍历 total.json，并从其他 JSON 文件（如 "train.json"）中取出对应数据
                typ = data["type"] + ".json" # 获取数据类型文件名
                if idx[typ] == len(all_data[typ]): # 到尾了
                    break 
                aim_data = all_data[typ][idx[typ]] # aim_data是所有data中对应的类型对应的访问到的idx的item
                assert aim_data["question"] == data["question"] # 确保question相同
                idx[typ] += 1 # 该type访问索引+=1
                total_data.append(aim_data) # 将数据加入 total_data
            return [["total.json", total_data]]
        
        # 不是total类型
        for filename in files:
            if filename != "total.json": # 将非total类型的其它data存入solve_dataset中
                with open(os.path.join(input_dir, filename), "r") as fin:
                    solve_dataset.append((filename, json.load(fin)))
        if data_type is None:
            return solve_dataset
        else: # data_type不是None
            data_type = data_type + ".json" # 转换为 data_type.json 格式
            if data_type not in [v[0] for v in solve_dataset]: # 判断该type是否在solve_dataset中
                raise ValueError(f"Invalid {data_type} in Dataset {data_name}")
            tmp = []
            for filename, dataset in solve_dataset:
                if filename == data_type:
                    tmp.append((filename, dataset))
            return tmp # 只返回匹配 data_type 的数据
    else:
        with open(os.path.join(input_dir, "total.json"), "r") as fin: # 只有一个文件
            solve_dataset.append(("total.json", json.load(fin))) # 直接添加total.json到solve_dataset
        return solve_dataset  # 返回solve_dataset
    

def get_model_path(model_name):
    if model_name == "llama3_8b": 
        return "/liuzyai04/thuir/LLM/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5_1.5b":
        return "/liuzyai04/thuir/tyc/LLM/Qwen2.5-1.5B-Instruct"
    elif model_name == "llama3.2_1b":
        return "/liuzyai04/thuir/tyc/LLM/Llama-3.2-1B-Instruct"
    else:
        return model_name


def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = dict(
        num_beams=1, # 贪心搜索
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    return model, tokenizer, generation_config

# -------------------------------- 仅用于 augmentation 时的生成----------------------------------------

def model_generate(prompt, model, tokenizer, generation_config): # 返回生成的结果
    messages = [{
        'role': 'user', 
        'content': prompt,
    }]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True
    )
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device) # 加 batch 维度，变成 [batch_size, sequence_length] 格式
    output = model.generate(
        input_ids, 
        attention_mask = torch.ones(input_ids.shape).to(model.device), # 所有 token 都可见
        **generation_config
    )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text

# ------------------------------------------------------------------------------------

def read_complete(filepath):
    try:
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return data, len(data)
    except:
        return [], 0

    
def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot: # 没有cot
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list: # 遇到 ., \n, , 就截断答案
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else: # 有cot
        if "the answer is" in pred: # 找到 the answer is 之后的部分（排除解释步骤）
            pred = pred[pred.find("the answer is") + len("the answer is"):]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list: # 同样根据stop_list截断
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip() 

    em = BaseDataset.exact_match_score( # 计算em和f1
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def predict(model, tokenizer, generation_config, question, with_cot, passages = None): # 用于真正的生成
    model.eval()
    input_ids = get_prompt( # 先调用get_prompt函数处理question和passages
        tokenizer, 
        question, 
        passages = passages, 
        with_cot = with_cot)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask = torch.ones(input_ids.shape).to(model.device),
            **generation_config)
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text