import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import WikiMultiHopQA, HotpotQA, CWQ, popQA
from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    tmp = parser.parse_args()
    with open(os.path.join(tmp.dir, "config.json"), "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    return args

def extract_answer(pred):
    return None
    if "the answer is" in pred:
        return pred.split("the answer is", 1)[-1].strip()
    return None

def generate_answer(tokenizer, model, question, context):
    question = question[question.find('Question:'):question.find('Answer:')]
    prompt = f"{question}\nPlease extract the answer to the question above from the context below.\n"
    prompt += f"Context: {context}\n\n"
    prompt += f"You should ONLY reply the exact answer to the question."

    print(prompt)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask = torch.ones_like(input_ids),
        max_new_tokens=10,
        do_sample=False
    )
    generated_ids = outputs[:, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text.strip()

def main():
    args = get_args()
    logger.info(f"{args}")
    
    datasets = {
        'cwq': CWQ,
        '2wikimultihopqa': WikiMultiHopQA,
        'hotpotqa': HotpotQA,
        'popqa': popQA
    }
    if args.dataset not in datasets:
        raise NotImplementedError
    
    data = datasets[args.dataset](args.data_path)
    data.format(fewshot=args.fewshot)
    
    dataset = {t["qid"]: (t["answer"], t.get("answer_id"), t.get("case")) for t in data.dataset}
    
    tokenizer, model = None, None
    if args.dataset in datasets:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, device_map="auto"
        )
    
    metrics = ["EM", "f1", "precision", "recall"]
    if "use_counter" not in args or args.use_counter==True:
        count_list = ["retrieve_count", "generate_count", "hallucinated_count", "token_count", "sentence_count"]
        metrics += count_list
    values = {m: [] for m in metrics}
    
    pred_out = open(f"{args.output_dir}/details.txt", "w")
    with open(os.path.join(args.output_dir, "output.jsonl"), "r") as fin:
        for line in tqdm(fin.readlines()):
            rd = json.loads(line)
            qid, pred = rd["qid"], rd["prediction"]
            ground_truth, ground_truth_id, case = dataset[qid]
            
            extracted_answer = extract_answer(pred)
            if extracted_answer is None and tokenizer and model:
                extracted_answer = generate_answer(tokenizer, model, case, pred)
            
            final_pred = data.get_real_prediction(extracted_answer)
            
            em_ret = data.exact_match_score(final_pred, ground_truth, ground_truth_id)
            f1_ret = data.f1_score(final_pred, ground_truth, ground_truth_id)
            
            values["EM"].append(em_ret["correct"])
            for k in f1_ret:
                values[k].append(f1_ret[k])
            
            if "use_counter" not in args or args.use_counter==True:
                for k in count_list:
                    values[k].append(rd[k])
            
            pred_out.write(json.dumps({
                "qid": qid,
                "final_pred": final_pred,
                "EM": str(em_ret["correct"]),
                "F1": str(f1_ret["f1"])
            }) + "\n")
    
    df = pd.DataFrame([[m, np.mean(values[m])] for m in metrics])
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False)
    
if __name__ == "__main__":
    main()
