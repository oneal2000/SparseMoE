from typing import Dict, List, Callable, Tuple, Union, Callable
import logging
import os
import json
import re
import glob
import string
import spacy
from collections import Counter
from tqdm import tqdm
import numpy as np
from datasets import Dataset
import pandas as pd

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BaseDataset:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        return {}

    @classmethod
    def normalize_answer(cls, s): # 转换成小写、去除冠词、去除标点、去除“多余的”空格
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score( # 计算EM分数
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth) 
        if ground_truth_id and isinstance(ground_truth_id, str): 
            ground_truths.update(cls.get_all_alias(ground_truth_id)) # 如果 ground_truth_id 存在，则使用 get_all_alias() 获取所有别名，并加入 ground_truths 集合

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths]) # 计算预测 prediction 是否与 ground_truths 中的任何一个完全匹配（经过 normalize_answer 规范化后）
        return {'correct': correct, 'incorrect': 1 - correct} # 匹配成功为 1，否则为 0

    @classmethod
    def f1_score(
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
            # 如果预测或正确答案是 "yes"、"no"、"noanswer"，则需要严格匹配，避免部分匹配带来的误差。
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth: 
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            
            # 对预测的和正确的分别进行空格分词，然后计算recall prec和f1
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

    def format(self, fewshot: int = 0): # 格式化数据，适配不同的输入模版，主要用于 Few-shot 
        def _format(
            example: Dict, # 里面通常有问题、推理过程、正确答案
            use_answer: bool = False, # 是否在最终的 query 中包含答案，True 时会拼接答案，False 则仅保留问题部分。
            input_template_func: Callable = None,
        ):
            q = example['question'] # 问题
            if 'cot' in example: # 如果有推理过程
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer'] # 答案

            query = input_template_func(q) # 如果有输入模版，就包装一下
            if use_answer: # 将cot和answer通过输入模版拼到query后面，中间有空格或者换行符分割
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'], #  原始问题文本。
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template), # 经过 _format 处理后的格式化问题和答案。
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else [] # 额外的上下文（如果存在）。
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template) # 仅格式化 question，不包含答案（use_answer=False）。
            # ctx
            example['demo'] = demo
            example['case'] = case
            return example
        
        self.dataset = self.dataset.map(_format_for_dataset) # 遍历数据集，将 _format_for_dataset 作用于每个示例。这样，每个数据样本都会增加：demo: few-shot 示例（如果 fewshot > 0）；case: 仅包含格式化的问题。
    
    def get_real_prediction(self, pred):
        if "the answer is" in pred:
            beg = pred.find("the answer is") + len("the answer is") + 1
            pred = pred[beg:]
        if pred.endswith("."):
            pred = pred[:-1]
        return pred


class WikiMultiHopQA(BaseDataset): # dataset格式：qid, question, answers
    examplars: List[Dict] = [
        {
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
            'answer': "19 June 2013",
        },
        {
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
            'answer': "no",
        },
        {
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
            'answer': "no",
        },
        {
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
            'answer': "Martin Hodge",
        },
        {
            'question': "When did the director of film Laughter In Hell die?",
            'cot': "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
            'answer': "August 25, 1963",
        },
        {
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
            'answer': "Genghis Khan",
        },
        {
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
            'answer': "Twenty Plus Two",
        },
        {
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
            'answer': "Prithvipati Shah",
        }
    ]
    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f"\nAnswer the following question through reasoning step-by-step, in the same format as examples above. Do not list steps explicitly. Avoid using phrases like'to ..., we need ...'. Instead, provide the factual reasoning directly.\nQuestion: {ques}\nAnswer:"
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str): 
        logger.info(f"Loading WikiMultiHopQA from {data_path}")
        with open(os.path.join(data_path, 'qid_list.txt'), 'r') as qid_fin:
            qid_list = [line.strip() for line in qid_fin]
        self.init_id_aliases(data_path)
        
        dataset = []
        with open(os.path.join(data_path, 'sample300.json'), 'r') as fin:
            js = json.load(fin)
            for example in tqdm(js):
                if example['_id'] not in qid_list:
                    continue
                qid = example['_id']
                question = example['question']
                ans = example['answer']
                if example['answer_id'] and example['answer_id'] in self.id_alias:
                    ans_aliases = self.id_alias[example['answer_id']]
                else:
                    ans_aliases = []
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': [ans] + ans_aliases,
                })
        self.dataset = Dataset.from_list(dataset)
        
    @classmethod
    def init_id_aliases(cls, data_path):
        cls.id_alias: Dict[str, List[str]] = {}
        with open(os.path.join(data_path, 'id_aliases.json'), 'r') as fin:
            for l in fin:
                l = json.loads(l)
                cls.id_alias[l['Q_id']] = l['aliases']

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        if ground_truth_id and ground_truth_id in cls.id_alias:
            return cls.id_alias[ground_truth_id]
        else:
            return []


class HotpotQA(BaseDataset): # dataset格式：qid, question, answer
    examplars: List[Dict] = [
        {
            'question': "Jeremy Theobald and Christopher Nolan share what profession?",
            'cot': "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer.",
            'answer': "producer",
        },
        {
            'question': "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
            'cot': "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau.",
            'answer': "The Phantom Hour.",
        },
        {
            'question': "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
            'cot': "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20.",
            'answer': "20",
        },
        {
            'question': "Were Lonny and Allure both founded in the 1990s?",
            'cot': "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s.", 
            'answer': "no",
        },
        {
            'question': "Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
            'cot': "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn.",
            'answer': "Scott Glenn",
        },
        {
            'question': "What was the 2014 population of the city where Lake Wales Medical Center is located?",
            'cot': "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140.",
            'answer': "15,140",
        },
        {
            'question': "Who was born first? Jan de Bont or Raoul Walsh?",
            'cot': "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first.",
            'answer': "Raoul Walsh",
        },
        {
            'question': "In what country was Lost Gravity manufactured?",
            'cot': "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company.",
            'answer': "Germany",
        },
        {
            'question': "Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?",
            'cot': "The debut album of the band \"Hot Hot Heat\" was \"Make Up the Breakdown\". The debut album of the band \"The Operation M.D.\" was \"We Have an Emergency\".",
            'answer': "The Operation M.D.",
        },
        {
            'question': "How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?",
            'cot': "The singer of \"A Girl Like Me\" singer is Rihanna. In the American Music Awards of 2012, Rihana won one award.",
            'answer': "one",
        },
        {
            'question': "The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?",
            'cot': "The actor that stars as Joe Proctor on the series \"Power\" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Thus, Turtle Assante's last name is Assante.",
            'answer': "Assante",
        },
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f"\nAnswer the following question through reasoning step-by-step, in the same format as examples above. Do not list steps explicitly. Avoid using phrases like'to ..., we need ...'. Instead, provide the factual reasoning directly.\nQuestion: {ques}\nAnswer:"
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading HotpotQA from {data_path}")
        with open(os.path.join(data_path, 'qid_list.txt'), 'r') as qid_fin:
            qid_list = [line.strip() for line in qid_fin]
        dataset = []
        with open(os.path.join(data_path, 'sample300.json'), "r") as fin:
            js = json.load(fin)
            for example in tqdm(js):
                if not example["_id"] in qid_list:
                    continue
                qid = example["_id"]
                question = example["question"]
                answer = example['answer']
                context = example['context']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                })
        self.dataset = Dataset.from_list(dataset)


class CWQ(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': "What is the name of the movie with the trailer featured at http://youtu.be/TZe1Qz2twbE and starring Alyson Stoner?",
            'cot': "The provided YouTube link corresponds to the trailer of the movie Step Up 3D. Alyson Stoner is one of the actors featured in the film. Thus, the name of the movie is Step Up 3D.",
            'answer': "Step Up 3D",
        },
        {
            'question': "Which of the 3 countries bordering Mexico has 320 as its ISO numeric?",
            'cot': "Mexico shares borders with three countries: the United States, Guatemala, and Belize. The ISO numeric code for the United States is 840, for Belize is 084, and for Guatemala is 320. Thus, the country with ISO numeric 320 is Guatemala.",
            'answer': "Guatemala",
        },
        {
            'question': "What language is spoken in the politician Mohammad Najibullah's country?",
            'cot': "Mohammad Najibullah was a politician from Afghanistan. The official languages of Afghanistan are Pashto and Dari. Thus, one of the languages spoken in his country is Pashto.",
            'answer': "Pashto language",
        },
        {
            'question': "Which attractions in Dubai are greater than 26 stories tall?",
            'cot': "Dubai is home to several tall buildings and attractions. One of the well-known attractions that exceeds 26 stories in height is the Burj Khalifa, which is the tallest building in the world. Additionally, Dubai Tower is another high-rise structure in Dubai. Thus, an attraction greater than 26 stories tall in Dubai is Dubai Tower.",
            'answer': "Dubai Tower",
        },
        {
            'question': "What is the major religion in the UK that believes in the deity 'Telangana Talli'?",
            'cot': "Telangana Talli is a deity associated with the Telangana region of India and is primarily worshipped by Hindus. Hinduism is a practiced religion in the UK due to its Indian diaspora. Thus, the major religion in the UK that believes in Telangana Talli is Hinduism.",
            'answer': "Hinduism",
        },
        {
            'question': "Find the state that has a Nene as the official symbol, what time zone is this state in?",
            'cot': "The Nene, also known as the Hawaiian goose, is the official state bird of Hawaii. Hawaii follows the Hawaii-Aleutian Time Zone. Thus, the state in question is in the Hawaii-Aleutian Time Zone.",
            'answer': "Hawaii-Aleutian Time Zone",
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f"\nAnswer the following question through reasoning step-by-step, in the same format as examples above. Do not list steps explicitly. Avoid using phrases like'to ..., we need ...'. Instead, provide the factual reasoning directly.\nQuestion: {ques}\nAnswer:"
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'
    
    def __init__(self, data_path: str):
        logger.info(f"Loading CWQ from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'ComplexWebQuestions_dev.json'), "r") as fin:
            js = json.load(fin)
            for example in tqdm(js):
                qid = example["ID"]
                question = example["question"]
                answer = []
                for ans in example["answers"]:
                    answer.append(ans["answer"])
                    answer.extend(ans["aliases"])
                answer = list(set(answer))
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                })
        self.dataset = Dataset.from_list(dataset)


class popQA(BaseDataset):
    examplars: List[Dict] = [ 
        {
            'question': "What is the capital of Louisiana?", # 802449
            'cot': "Louisiana is a state in the United States. The capital city of Louisiana is Baton Rouge, which serves as the administrative and political center of the state.",
            'answer': "Baton Rouge",
        },
        {
            'question': "Who is the mother of Margaret of Sicily?", # 5502218
            'cot': "Margaret of Sicily was a historical figure. Her mother was Isabella of England, who was married to Emperor Frederick II and was the queen consort of Sicily.",
            'answer': 'Isabella of England',
        },
        {
            'question': "Who is the author of Carmelite Rule of St. Albert?", # 2624911
            'cot': "The Carmelite Rule of St. Albert was written by Albert of Vercelli, also known as Albert Avogadro, who was the Latin Patriarch of Jerusalem and formulated the rule for the Carmelite Order.",
            'answer': 'Albert of Vercelli',
        },
        {
            'question': "What sport does Paul Hoffman play?", # 953373
            'cot': "Paul Hoffman is known for his involvement in sports. Specifically, he played basketball, making him recognized in the sport.",
            'answer': 'Basketball',
        },
        {
            'question': "Who was the director of Les Palmes de M. Schutz?", # 6515850
            'cot': "Les Palmes de M. Schutz is a film directed by Claude Pinoteau, a French filmmaker known for his contributions to cinema.",
            'answer': 'Claude Pinoteau',
        },
        {
            'question': "What genre is Simon Le Bon?", # 3059089
            'cot': "Simon Le Bon is a musician and the lead singer of the band Duran Duran. The genre of music he is associated with is pop music.",
            'answer': 'pop music',
        } 
    ]

    
    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f"\nAnswer the following question through reasoning step-by-step, in the same format as examples above. Do not list steps explicitly. Avoid using phrases like'to ..., we need ...'. Instead, provide the factual reasoning directly.\nQuestion: {ques}\nAnswer:"
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading popQA from {data_path}")
        data_path = os.path.join(data_path, "sample300.tsv")
        dataset = pd.read_csv(data_path, sep="\t")
        new_dataset = [] # 存储转换后的数据
        
        def to_utf8(text): # 数据集里面有些“国旗”是非ascii的，需要手动转换一下。
            if isinstance(text, str):
                return text.encode("utf-8", "ignore").decode("utf-8")  # 忽略无法编码的字符
            return text
        
        for did in range(len(dataset)):
            data = dataset.iloc[did]
            question = to_utf8(data["question"])
            answer = [to_utf8(data["obj"])] + [to_utf8(x) for x in eval(data["o_aliases"])]
            qid = str(data['id'])
            val = {
                "qid": qid, # 这里改成了用id字段而不是遍历序号作为qid
                "question": question, 
                "answer": answer,
            }
            new_dataset.append(val)
        self.dataset = Dataset.from_list(new_dataset)


if __name__ == "__main__":
    # 设定数据路径
    data_path = "../data/popqa"  # 替换成你的实际路径
    # 创建数据集对象
    try:
        dataset = popQA(data_path)
        logger.info(f"成功加载数据集，共 {len(dataset.dataset)} 条数据")

        # 打印部分示例
        for i in range(min(50, len(dataset.dataset))):
            print(dataset.dataset[i])

    except Exception as e:
        logger.error(f"加载数据集时出错: {e}", exc_info=True)
