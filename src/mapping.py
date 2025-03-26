import os
import json

results_dir = "../result/3.27"
qid_to_rel_psgs = {}

for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    subfolders = [sf for sf in os.listdir(folder_path) if sf.isdigit()]
    if not subfolders:
        continue
    
    max_subfolder = max(subfolders, key=int)
    output_file = os.path.join(folder_path, max_subfolder, "output.jsonl")
    print(output_file)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                qid = f"{folder}_{str(data['qid'])}" # 因为popqa是以int作为qid的
                rel_psgs = data["rel_psgs"]
                
                qid_to_rel_psgs[qid] = rel_psgs

tot_psgs_num = 0
# 对list去重
for qid in qid_to_rel_psgs:
    qid_to_rel_psgs[qid] = list(set(qid_to_rel_psgs[qid]))
    tot_psgs_num += len(qid_to_rel_psgs[qid])

print(len(qid_to_rel_psgs), tot_psgs_num) # 1200, 9323
# 将结果写入 JSON 文件
output_json_file = "mapping_results.json"
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(qid_to_rel_psgs, f, ensure_ascii=False, indent=4)