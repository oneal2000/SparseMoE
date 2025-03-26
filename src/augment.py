import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm

from retrieve.retriever import bm25_retrieve
from utils import get_model, model_generate
from root_dir_path import ROOT_DIR

random.seed(42)


def load_popqa(data_path):
    data_path = os.path.join(data_path, "sample300.tsv")
    dataset = pd.read_csv(data_path, sep="\t")
    new_dataset = [] # 存储转换后的数据
    for did in range(len(dataset)):
        data = dataset.iloc[did] # 通过 iloc 按索引获取 DataFrame 的第 did 行数据
        question = data["question"]
        answer = [data["obj"]] + eval(data["o_aliases"]) # 先将 obj（标准答案）作为 answer 列表的第一个元素。o_aliases 是 JSON 格式的字符串，通过 eval() 解析成 Python 列表
        val = {
            "qid": str(data['id']), 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    return {"total": new_dataset}


def load_cwq(data_path):
    data_path = os.path.join(data_path, "sample300.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    for did, data in enumerate(dataset):
        question = data["question"]
        answer = []
        for ans in data["answers"]:
            answer.append(ans["answer"])
            answer.extend(ans["aliases"])
        answer = list(set(answer))
        val = {
            "qid": data["ID"],
            "test_id": did, 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    ret = {"total": new_dataset}
    return ret


def load_2wikimultihopqa(data_path):
    with open(os.path.join(data_path, "sample300.json"), "r") as fin:
        dataset = json.load(fin)
    with open(os.path.join(data_path, "id_aliases.json"), "r") as fin: # 实体的所有别名
        aliases = dict()
        for li in fin:
            t = json.loads(li)
            aliases[t["Q_id"]] = t["aliases"]
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        ans_id = data["answer_id"]
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": aliases[ans_id] if ans_id else data["answer"] # 如果 answer_id 存在，就从 aliases 里面查找别名（如果该实体有多个别名，则返回所有别名）。
        }
        golden_passages = []
        contexts = {name: " ".join(sents) for name, sents in data["context"]} # 存储文档标题和对应的句子
        for fact_name, _sent_id in data["supporting_facts"]: # 提供了支持问题推理的关键句子
            psg = contexts[fact_name]
            golden_passages.append(psg)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val) # 按 type 分类存储数据
    ret = {"total": new_dataset} # 存储所有数据
    ret.update(type_to_dataset) # 把分类存储的数据合并到 ret
    return ret


def load_hotpotqa(data_path):
    data_path = os.path.join(data_path, "sample300.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": data["answer"]
        }
        tmp = []
        contexts = {name: "".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            tmp.append(psg)
        golden_passages = []
        for p in tmp:
            if p not in golden_passages:
                golden_passages.append(p)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_default_format_data(data_path):
    filename = data_path.split("/")[-1] # 通过 split("/") 获取路径中的最后一部分，即文件名
    assert filename.endswith(".json"), f"Need json data: {data_path}" # 文件必须是 .json 格式
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    for did, data in enumerate(dataset):
        assert "question" in data, f"\"question\" not in data, {data_path}" # 确保 data 字典中存在 question 字段
        question = data["question"]
        assert type(question) == str, f"\"question\": {question} should be a string" # 确保 question 的值是字符串（str）
        assert "answer" in data, f"\"answer\" not in data, {data_path}" # 确保 answer 字段存在
        answer = data["answer"] # 是一个字符串or是list 且 answer 中所有元素均为字符串
        assert type(answer) == str or \
               (type(answer) == list and (not any(type(a) != str for a in answer))), \
               f"\"answer\": {answer} should be a string or a list[str]" 
        data["test_id"] = did # 新增 test_id 字段，赋值为当前数据的索引编号 did
    return {filename: dataset} # 键为 filename（文件名），值为校验后的 dataset


def get_rewrite(passage, model_name, model=None, tokenizer=None, generation_config=None):
    rewrite_prompt = "Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.\n{passage}"
    return model_generate(rewrite_prompt.format(passage=passage), model, tokenizer, generation_config)

# 重写的模版
qa_prompt_template = "I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least three elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"

def fix_qa(qa): # 检查并修正由模型生成的 QA 列表
    if isinstance(qa, list): # 先检查 qa 是否是 列表
        if len(qa) >= 3: # 限制 qa 列表最多包含 3 个元素
            qa = qa[:3]
            for data in qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data: # 检查每个 data 是否包含必要字段
                    return False, qa
                if isinstance(data["answer"], list): # 保证answer是字符串类型
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa

def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None): 
    # 调用 语言模型 生成 QA（问题-答案）对，并进行格式修正，确保其符合 JSON 格式并通过 fix_qa 校验。

    def fix_json(output):
        if model_name == "llama3.2_1b":
            output = output[output.find("["):] # 只保留从 [ 开始的部分
            if output.endswith(","): # 如果结尾是 逗号 ,，去掉
                output = output[:-1]
            if not output.endswith("]"): # 如果结尾缺少 ]，补上
                output += "]"
        elif model_name == "llama3_8b":
            if "[" in output: # 确保只保留 [ ... ] 之间的内容
                output = output[output.find("["):] 
            if "]" in output:
                output = output[:output.find("]")+1] # 截取到 ] 结尾，去除无关字符
        return output

    try_times = 100
    prompt = qa_prompt_template.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output)
        try:
            qa = json.loads(output) # 尝试 解析 JSON 并转换为 Python 结构
            ret, qa = fix_qa(qa) # 调用 fix_qa() 进行数据校验与修正
            if ret: # 如果 fix_qa 通过（ret=True），直接返回 qa
                return qa
        except:
            try_times -= 1 # 如果 JSON 解析失败（格式错误），减少 try_times 并重试
    return output # 如果尝试 100 次仍失败，返回 未修正的 output 作为最终结果

def read_mapping(mapping_path="/liuzyai04/thuir/yuebaoqing/vsmoe/src/mapping_results.json"):
    with open(mapping_path, 'r') as file:
        mapping = json.load(file)
    return mapping

def main(args):
    output_dir = os.path.join(ROOT_DIR, "3.27", args.dataset, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    mapping = read_mapping()

    print("### Loading dataset ###") # 检查是否存在针对指定数据集 (args.dataset) 的加载函数（例如 load_dataset_name）。如果存在，就使用该函数；如果没有，则使用默认的加载函数 妙！
    if f"load_{args.dataset}" in globals():
        load_func = globals()[f"load_{args.dataset}"]
    else:
        print('error! ')
        load_func = globals()["load_default_format_data"]
    load_dataset = load_func(args.data_path)
    if len(load_dataset) == 1:
        solve_dataset = load_dataset # 如果加载的数据集只有一项，则直接将 solve_dataset 设为 load_dataset
    else:
        solve_dataset = {k: v for k, v in load_dataset.items() if k != "total"} # 如果数据集中有多项，就从 load_dataset 中去除 total 项，然后将剩余的部分赋给 solve_dataset
        with open(os.path.join(output_dir, "total.json"), "w") as fout:
            json.dump(load_dataset["total"][:args.sample], fout, indent=4) # 将 total 部分的数据保存到 total.json 文件中，并仅保存 args.sample 数量的数据
    
    model, tokenizer, _ = get_model(args.model_name)
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.7,
        top_k=50,
    )

    for filename, dataset in solve_dataset.items():
        print(f"### Solving {filename} ###")
        output_file = os.path.join(
            output_dir, 
            filename if filename.endswith(".json") else filename + ".json"
        )
        ret = []
        dataset = dataset[:args.sample] # 只取了dataset的前args.sample个，而非使用完整的数据集
        # pbar = tqdm(total = args.sample * args.topk) # 初始化进度条 pbar，用于显示处理进度，总数为 args.sample * args.topk
        for data in tqdm(dataset,desc="traversing"):
            # passages = bm25_retrieve(data["question"], topk=args.topk) # 根据 question 字段使用 bm25_retrieve 函数检索相关段落。检索的数量是 args.topk+10，多检索一些段落以备筛选
            passages = mapping[f"{args.model_name}_{args.dataset}_{str(data['qid'])}"] # 之所以没加上用全文搜出来的passages，是因为那些已经encode过了，即使需要用到，也直接拼即可。
            final_passages = [] # 存放最终筛选的段落
            data["augment"] = [] # 存放增强后的数据（包括重写的段落和 QA 对）
            for psg in passages: # passages就是召回的dpr里面的passage，一般3个
                val = { 
                    "pid": len(final_passages),
                    "passage": psg, 
                    f"{args.model_name}_rewrite": get_rewrite(psg, args.model_name, model, tokenizer, generation_config)
                }
                qa = get_qa(psg, args.model_name, model, tokenizer, generation_config)
                if fix_qa(qa)[0] == False: # skip error passage
                    continue
                val[f"{args.model_name}_qa"] = qa
                data["augment"].append(val)
                final_passages.append(psg)
                # pbar.update(1) # pbar += 1
                ##### 我把下面这个删了 #####
                # if len(data["augment"]) == args.topk: # 如果 data["augment"] 达到 args.topk
                #     break
                #########################
                
            data["passages"] = final_passages
            ret.append(data)
        with open(output_file, "w") as fout:
            json.dump(ret, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--topk", type=int, default=3) 
    args = parser.parse_args()
    print(args)
    main(args)