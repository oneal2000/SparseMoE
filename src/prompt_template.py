import os
from root_dir_path import ROOT_DIR

current_dataset = None
fewshot = None
fewshot_path = os.path.join(ROOT_DIR, "src", "fewshot")

USER_PROMPT = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Question: {question}"

USER_PROMPT_WITH_COT = "You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.\n\
Here are some examples about how to answer the questions.\n\
{fewshot}\
Here are some reference.\n\
{passages}\n\n\
Let's think step by step. Answer the questions in the same format as above.\n\
Question: {question}"

ASSISTANT_PROMPT = "The answer is {answer}"
ASSISTANT_PROMPT_WITH_COT = "Answer: {answer}"

def _get_prompt(question, passages=None, answer=None):
    question = question.strip()
    if not question.endswith('?'):
        question = question.strip() + '?' # # 如果 `question` 末尾没有 `?`，添加 `?`
    elif question.endswith(' ?'):
        question = (question[:-1]).strip() + '?' # # 如果 `question` 末尾是 `' ?'`，去掉空格，确保整洁
     
    if passages and not isinstance(passages, list): # 如果 `passages` 不是列表，则转换为列表
        passages = [passages]
    
    if answer is None: # 如果 `answer` 为空，则设为空字符串
        answer = ""
    else:
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += "."  # 确保 `answer` 以 `.` 结尾
    return question, passages, answer


def get_fewshot(dataset): # 修改全局变量fewshot，则fewshot无需返回也可在不同函数间共享
    import json
    global current_dataset
    global fewshot
    # assert current_dataset is None
    if dataset.endswith("_golden"): # 如果 dataset 以 "_golden" 结尾，则去掉该后缀，获取原始数据集名称
        dataset = dataset.split("_golden")[0]
    current_dataset = dataset # 标记当前数据集
    with open(os.path.join(fewshot_path, dataset + ".json"), "r") as fin:
        tmp = json.load(fin)
    fewshot = ""
    for data in tmp: # tmp是一个 Python 列表，其中每个元素是一个 字典，包含 "question" 和 "answer" 键
        q = data["question"]
        a = data["answer"]
        fewshot += f"Question: {q}\nAnswer: {a}\n\n"


def get_prompt(tokenizer, question, passages=None, answer=None, with_cot=False):
    question, passages, answer = _get_prompt(question, passages, answer) # 格式化预处理
    contexts = ""
    if passages: # 格式化passages
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"
    if not with_cot: # 没有cot
        user_content = USER_PROMPT.format(question=question, passages=contexts)
        assistant_content = ASSISTANT_PROMPT.format(answer=answer)
    else: # 有cot
        assert fewshot is not None
        user_content = USER_PROMPT_WITH_COT.format(question=question, passages=contexts, fewshot=fewshot)
        assistant_content = ASSISTANT_PROMPT_WITH_COT.format(answer=answer)

    messages = [{
        "role": "user",
        "content": user_content,
    }]

    inputs = tokenizer.apply_chat_template( # 给输入套上chat_template
        messages, # 里面是user_content
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False) # add_special_tokens=False避免重复添加 BOS/CLS 等标记
    return inputs