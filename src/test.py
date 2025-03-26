# import torch
# from transformers import pipeline

# model_id = "/liuzyai04/thuir/tyc/LLM/Llama-3.2-1B-Instruct"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# prompt = f"""
# Example[1]: Question: When did the director of film Hypocrite (Film) die?
# Answer: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.
# Example[2]: Question: Are both Kurram Garhi and Trojkrsti located in the same country?
# Answer: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.
# Example[3]: Question: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?
# Answer: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two. So the answer is Twenty Plus Two.
# Example[4]: Question: Who is the grandchild of Krishna Shah (Nepalese Royal)?
# Answer: Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah. So the answer is Prithvipati Shah.
# Relevant references:
# [1] Sleeping through the night: are extended-release formulations the answer?
# To provide an overview of insomnia, including identification and current treatments, as well as review the efficacy and safety of extended-release sleep medication. Published clinical research and review articles, DSM-IV criteria, and clinical trials. Insomnia is a highly prevalent and debilitating sleep disorder, which may present with one or more of the following symptoms: difficulty initiating sleep, difficulty maintaining sleep, or waking too early without being able to return to sleep. Difficulty maintaining sleep throughout the night is the most common symptom of insomnia. The recently approved nonbenzodiazepine hypnotic, zolpidem extended-release, can be taken as long as medically necessary to improve sleep-onset and reduce sleep-maintenance difficulties in insomnia patients, without negatively affecting next-day functioning. Insomnia continues to be an underdiagnosed and undertreated disorder. The nurse practitioner, through routine inquiry about patient sleep habits and consideration of the appropriate treatment of insomnia, can help restore the quality of life of patients experiencing the negative consequences of insomnia.
# [2] Sleeping through the night: are extended-release formulations the answer?
# To provide an overview of insomnia, including identification and current treatments, as well as review the efficacy and safety of extended-release sleep medication. Published clinical research and review articles, DSM-IV criteria, and clinical trials. Insomnia is a highly prevalent and debilitating sleep disorder, which may present with one or more of the following symptoms: difficulty initiating sleep, difficulty maintaining sleep, or waking too early without being able to return to sleep. Difficulty maintaining sleep throughout the night is the most common symptom of insomnia. The recently approved nonbenzodiazepine hypnotic, zolpidem extended-release, can be taken as long as medically necessary to improve sleep-onset and reduce sleep-maintenance difficulties in insomnia patients, without negatively affecting next-day functioning. Insomnia continues to be an underdiagnosed and undertreated disorder. The nurse practitioner, through routine inquiry about patient sleep habits and consideration of the appropriate treatment of insomnia, can help restore the quality of life of patients experiencing the negative consequences of insomnia.
# [3] Answering the short-answer question paper.
# The short-answer question paper of the MRCPsych Part Two examination needs to be tackled in a different way from the other papers. This article provides guidance for the candidate in how to approach and to answer the questions.
# Answer the following question by reasoning step-by-step, following the examples above.
# Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
# """

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt},
# ]
# outputs = pipe(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1]['content'])


import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_id =  "/liuzyai04/thuir/tyc/LLM/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


prompt = f"""
Example[1]: Question: When did the director of film Hypocrite (Film) die?
Answer: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.
Example[2]: Question: Are both Kurram Garhi and Trojkrsti located in the same country?
Answer: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.
Example[3]: Question: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?
Answer: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two. So the answer is Twenty Plus Two.
Example[4]: Question: Who is the grandchild of Krishna Shah (Nepalese Royal)?
Answer: Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah. So the answer is Prithvipati Shah.
Relevant references:
[1] Sleeping through the night: are extended-release formulations the answer?
To provide an overview of insomnia, including identification and current treatments, as well as review the efficacy and safety of extended-release sleep medication. Published clinical research and review articles, DSM-IV criteria, and clinical trials. Insomnia is a highly prevalent and debilitating sleep disorder, which may present with one or more of the following symptoms: difficulty initiating sleep, difficulty maintaining sleep, or waking too early without being able to return to sleep. Difficulty maintaining sleep throughout the night is the most common symptom of insomnia. The recently approved nonbenzodiazepine hypnotic, zolpidem extended-release, can be taken as long as medically necessary to improve sleep-onset and reduce sleep-maintenance difficulties in insomnia patients, without negatively affecting next-day functioning. Insomnia continues to be an underdiagnosed and undertreated disorder. The nurse practitioner, through routine inquiry about patient sleep habits and consideration of the appropriate treatment of insomnia, can help restore the quality of life of patients experiencing the negative consequences of insomnia.
[2] Sleeping through the night: are extended-release formulations the answer?
To provide an overview of insomnia, including identification and current treatments, as well as review the efficacy and safety of extended-release sleep medication. Published clinical research and review articles, DSM-IV criteria, and clinical trials. Insomnia is a highly prevalent and debilitating sleep disorder, which may present with one or more of the following symptoms: difficulty initiating sleep, difficulty maintaining sleep, or waking too early without being able to return to sleep. Difficulty maintaining sleep throughout the night is the most common symptom of insomnia. The recently approved nonbenzodiazepine hypnotic, zolpidem extended-release, can be taken as long as medically necessary to improve sleep-onset and reduce sleep-maintenance difficulties in insomnia patients, without negatively affecting next-day functioning. Insomnia continues to be an underdiagnosed and undertreated disorder. The nurse practitioner, through routine inquiry about patient sleep habits and consideration of the appropriate treatment of insomnia, can help restore the quality of life of patients experiencing the negative consequences of insomnia.
[3] Answering the short-answer question paper.
The short-answer question paper of the MRCPsych Part Two examination needs to be tackled in a different way from the other papers. This article provides guidance for the candidate in how to approach and to answer the questions.
Answer the following question by reasoning step-by-step, following the examples above.
Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
"""
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": prompt},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

output = model.generate(
    input_ids,
    max_new_tokens=256,
    return_dict_in_generate=True,
    attention_mask=torch.ones_like(input_ids),
    output_scores=True
)

generated_ids = output.sequences[:, input_ids.shape[1]:]
scores = output.scores

# 解码生成的文本
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 计算每个 token 的 log probability
import torch.nn.functional as F
logprobs = []
for i, logits in enumerate(scores):
    probs = F.log_softmax(logits, dim=-1)
    token_log_prob = probs[0, generated_ids[0, i]].item()
    logprobs.append(torch.tensor(token_log_prob)) 
# logprobs = [p[1].numpy() for p in log_probs]
logprobs = torch.stack(logprobs).cpu().numpy()
print("Generated text:")
print(generated_text)
# print("\nToken Log Probs:")
# for token, log_prob in log_probs:
#     print(f"{token}: {log_prob}")
