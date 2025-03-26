import numpy as np
import logging
import spacy
import torch
import torch.nn.functional as F
from math import exp
from scipy.special import softmax
from retriever import BM25
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import argparse
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class LlamaGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None: # 确保 pad_token 存在，否则将其设为 eos_token
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
             
    def generate(self, input_text, max_length, return_logprobs=False):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
        
        if return_logprobs:  # 如果要返回对数概率的话
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                attention_mask = torch.ones_like(input_ids),
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False  # 关闭随机采样，保证输出一致
            )
            
            transition_scores = self.model.compute_transition_scores( # 计算 transition_scores（对数概率）
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            input_length = input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs] # 转换 logprobs 为 NumPy 数组 并返回
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs  # 返回文本、tokens 和 logprobs
        
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask = torch.ones_like(input_ids),
                max_new_tokens=max_length,
                do_sample=False
            )
            generated_ids = outputs[:, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text, None, None  # 只返回文本
        

class Counter:
    def __init__(self):
        self.retrieve = 0 # 召回计数
        self.generate = 0 # 生成计数
        self.hallucinated = 0 # 幻觉计数
        self.token = 0 # token计数
        self.sentence = 0 # 句子计数

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items(): # 使用 setattr 方法将其存储到 self 对象中，使得 self 拥有所有的初始化参数
            setattr(self, k, v)
        self.generator = LlamaGenerator(self.model_name_or_path) # generator用默认的
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, # 全用的wiki
                    engine = "elasticsearch",
                )
        # print(self.generate_max_length)
        self.counter = Counter() # 初始化Counter，记录 retrieve（检索） 和 generate（生成） 的调用次数。

    def retrieve(self, query, topk=1, max_query_length=64): # 返回topk的n个文档
        self.counter.retrieve += 1 # 每次调用 retrieve，增加 self.counter.retrieve 计数，表示执行了一次检索
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve( # 使用 BM25 进行检索，返回 topk 个文档，并取出topk个文档作为检索结果
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0] 
        else:
            raise NotImplementedError

    def inference(self, question, demo, case): # 不召回
        assert self.query_formulation == "direct" # 只支持direct方式的查询
        if demo and demo != "":
            prompt = "".join([d["case"]+"\n" for d in demo]) + case # 将所有示例拼接在一起，并追加 case 作为最终输入。
        else:
            prompt = case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length) # 未返回logprobs
        if self.use_counter == True: # 如果 use_counter 设定为 True，则记录 generate 的调用次数，并统计生成的 token 数。
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
    
    
class TokenRAG(BasicRAG):
    def __init__(self, args): # 完全继承了BasicRAG的init
        super().__init__(args)
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences: # 对每个句子，使用 spaCy 的命名实体识别（NER）提取实体 (doc.ents)，并将每个句子的实体存储在 entity 列表中
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text) # 创建一个 belonging 列表，用于追踪 tokens 列表中的每个标记在原始文本中的位置。通过遍历 tokens 并计算它们在文本中的位置，来建立这个映射
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok)
            if apr == -1:
                break
            apr += pos
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = [] # 构建 entity_intv 列表，记录每个句子中每个实体的起始和结束标记索引。对于每个句子，首先找出实体在文本中的位置，然后记录它在标记列表中的起止索引
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = [] # 对每个实体，计算其概率，表示该实体是否有可能是幻觉。通过查看构成该实体的标记的对数概率来计算这个概率
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        rel_psgs = set()
        text = ""
        while True:
            old_len = len(text) # 旧的len是text的len
            prompt = f"You should answer my question through reasoning step-by-step. Here are {len(demo)} examples:\n"
            prompt += "".join([f"Example[{idx+1}]: {d['case']}\n" for idx, d in enumerate(demo)]) # 组合所有 demo 作为前缀
            prompt += '\n' + case + " " + text # 加上 case(问题) 和已有的生成文本
            # prompt += "\nPlease summarize your answer by several words at the end of your response in the format of 'the answer is ...'\n"
            # print('-'*100)
            # print('initial prompt:\n', prompt)
            # print('-'*100)
            new_text, tokens, logprobs = self.generator.generate( # 新生成的文本，tokens和logprobs
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            if self.use_counter == True: # 如果启用了 self.use_counter，记录生成的文本信息（统计生成次数）
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs) # 使用 modifier 方法检测并修正幻觉部分。
            if not hallucination:
                print ("no hallu")
                text = text.strip() + " " + new_text.strip() # 如果没有检测到幻觉，则将新的文本加入 text 中
                print('@'*100)
                print('text: ', text)
                print('@'*100)
                break
            else: # 如果检测到幻觉，则需要基于修正后的文本重新进行检索，生成新的查询并获取更多文档。
                if self.query_formulation == "direct": # 直接替换法进行检索，只考虑modifier之后的curr，不考虑之前其它生成的内容
                    retrieve_question = curr.replace("[xxx]", "[]") # 我把这块改成中括号了，要不然prompt里面不知道哪个地方是空的
                elif self.query_formulation == "forward_all":
                    curr = curr.replace("[xxx]", "[]")
                    # tmp_all = [question, text, ptext] # 用空格连接起来问题、text和ptext
                    tmp_all = [question, text, curr] # I change this 
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented
                print('='*100)
                print(f"retrieve_question: {retrieve_question}")
                print('='*100)
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk) # 召回回来的docs
                print(docs)
                # =================== #
                for doc in docs:
                    rel_psgs.add(doc)
                # =================== #
                prompt = f"You should answer my question through reasoning step-by-step. Here are {len(demo)} examples:\n"
                prompt += "".join([f"Example[{idx+1}]: {d['case']}\n" for idx, d in enumerate(demo)])
                prompt += "\nGiven the following information about this question:\n"
                for i, doc in enumerate(docs): # 加上召回回来的docs作为“context”
                    prompt += f"[{i+1}] {doc}\n"
                prompt += '\n' + case + " " + text + " " + ptext.strip() # case + text + ptext 作为已有信息
                # prompt += "\nPlease summarize your answer by several words at the end of your response in the format of 'the answer is ...'\n"
                print('*' * 100)
                print(prompt)
                print('*' * 100)
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length) # 这次新生成的，只考虑新生成的文本，不管tokens和logprobs了
                # 其实我要做的应该很简单，就是在这第二步generate处，调用prag的方法，而不是像flare这样retrieve回来docs拼入prompt。第一步generate应该无需修改。
                if self.use_counter == True: # 计数
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                # print('^'*100)
                # print('new_text:', new_text)
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip() # 拼接
                
                print('@'*100)
                print('text: ', text)
                print('@'*100)
                
                break # 这里做一个简化处理，最多执行幻觉判断一次后就停止。
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text, rel_psgs
       

def test_retriever(args):
    # 创建 BasicRAG 实例
    rag = BasicRAG(args)

    # 测试 retrieve 方法
    query = "What is George Rankin's occupation?"
    retrieved_doc = rag.retrieve(query, topk=3)
    for doc in retrieved_doc:
        print('='*80)
        print(doc)
        
        
def test_modifier(args):
    # 模拟输入数据
    BG = LlamaGenerator(args.model_name_or_path)
    input_text = "Who is Bill Clinton?"
    text, tokens, logprobs = BG.generate(input_text, 100, True)
    # print('text: ', text, '\n', 'tokens: ', tokens, '\n', 'logprobs: ', logprobs, '\n')
    print(f"text:{text}")
    
    model = EntityRAG(args)

    prev, curr, hallucinated = model.modifier(text, tokens, logprobs)

    print("\nOriginal Text:")
    print(text)
    if hallucinated:
        print("\nModified Text:")
        print(prev + " ---- " + curr)
    else:
        print("\nNo hallucination detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/liuzyai04/thuir/tyc/LLM/Llama-3.2-1B-Instruct")
    parser.add_argument("--retriever", type=str, default="BM25")  # 指定使用 BM25
    parser.add_argument("--es_index_name", type=str, default="wiki")  # 可选，Elasticsearch 索引名
    parser.add_argument("--sentence_solver", type=str, default="avg")
    parser.add_argument("--entity_solver", type=str, default="avg")
    parser.add_argument("--hallucination_threshold", type=int, default=0.1)
    args = parser.parse_args()  # 适用于 Jupyter 或直接运行
    # test_modifier(args)
    
    test_retriever(args)


