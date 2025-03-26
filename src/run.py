import os
import subprocess
import glob
import time
from multiprocessing import Pool

def run_config(args):
    config_path, gpu_id = args
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py -c {config_path}"
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    config_dir = "../configs"
    config_dirs = [os.path.join(config_dir, subdir) for subdir in os.listdir(config_dir) 
               if os.path.isdir(os.path.join(config_dir, subdir))]
    json_files = []
    for sub_dir in config_dirs:
        for target in ["hotpotqa", "2wikimultihopqa"]:
            target_dir = os.path.join(sub_dir, target)
            if os.path.isdir(target_dir):
                json_files.extend(glob.glob(os.path.join(target_dir, "*.json")))
                
    print(json_files)
    num_gpus = 6  # 你有 4 张显卡
    pool = Pool(processes=num_gpus)
    
    tasks = [(config, i % num_gpus) for i, config in enumerate(json_files)]
    
    pool.map(run_config, tasks)
    
    pool.close()
    pool.join()
    print("All processes finished.")
