# Sparse and Extendable MoE



## Overview

**Sparse and Extendable Mixture of Experts (MoE)** is a novel approach to enhancing large language models (LLMs) with dynamic expert integration during text generation. The core idea is to selectively retrieve and incorporate additional experts based on the uncertainty of generated tokens, ensuring more accurate and contextually relevant responses.


### Key Mechanism

- **Uncertainty-Based Retrieval**: During text generation, if a token's uncertainty exceeds a predefined threshold, the system triggers a retrieval step. The most recently generated sentence is used as a query to search for relevant documents via Elastic Search.
- **Expert Augmentation**: Each retrieved document corresponds to a pre-trained expert representation. The identified experts are dynamically integrated into the LLM to refine subsequent text generation.
- **Adaptive Expert Switching**: If new uncertainty tokens appear later in the generation, the retrieval and expert integration process is repeated, replacing the previous experts with more relevant ones.


### Components
- End-to-end implementation of the Sparse and Extendable MoE pipeline.
- Preprocessed benchmark datasets for experiments and scripts for customizing and adding new datasets.
- One-click bash scripts to reproduce our experiment results


This approach ensures that LLMs remain both **sparse** (using minimal resources when certainty is high) and **extendable** (adapting to new information dynamically), providing more reliable and informed responses in real-time applications.



**If you find our project interesting or helpful, we would sincerely appreciate it if you could give us a star! Your support is a tremendous encouragement to us!**


## Reproduce Paper Results  

This section demonstrates how to test the performance of **Sparse and Extendable MoE** on various QA datasets. The method dynamically incorporates retrieval-augmented expert modules into the model when generating responses, enhancing the LLM’s adaptability and accuracy.  

### Steps to Run Sparse and Extendable MoE  

1. **Run the Retrieval Module**  
   - Identify **uncertain tokens** in the generated text.  
   - Use the **preceding sentence** as a query to retrieve relevant documents from **Elastic Search**.  
   - Attach these documents to their corresponding questions.
   - Retrieve the **expert parameters** associated with the documents.  

2. **Integrate Experts into the LLM**  
    Before integrating experts, complete the following 2 steps:

    - Perform Data Augmentation: Enhanceing the dataset by transforming documents into an enriched form suitable for parameterization.

    - Train Parametric Representations: Involving training LoRA parameters to create expert representations associated with documents.

    Once these procedures are completed, proceed with expert integration:

    - Insert the retrieved **expert parameters** into the base LLM to enrich its knowledge.  
    - Incorporate the retrieved expert parameters into the LLM.
    - Use the updated model for further generation, dynamically adapting to uncertain tokens as needed.


Since you have understood the principles outlined above, the following instructions will guide you through the step-by-step implementation of Sparse and Extendable MoE.

#### Install Environment

```
conda create -n semoe python=3.10.4
conda activate semoe
pip install torch==2.1.0
pip install -r requirements.txt
```

Please change the `ROOT_DIR` variable in `src/root_dir_path.py` to the folder address where you store SEMOE.

#### Self-Augmentation

You can directly use the pre-augmented data file `data_aug.tar.gz`. To extract it, run the command `tar -xzvf data_aug.tar.gz` in your terminal.

If you want to perform data augmentation yourself, please process it as follows.

#### Prepare BM25 for retrieval

1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

2. Use Elasticsearch to index the Wikipedia dump

```bash
cd data
wget -O elasticsearch-8.15.5.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.5-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.5.tar.gz
rm elasticsearch-8.15.5.tar.gz 
cd elasticsearch-8.15.5
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

#### Download dataset

For 2WikiMultihopQA:

Download the [2WikiMultihopQA](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository <https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1>. Unzip it and move the folder to `data/2wikimultihopqa`.

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

For PopQA:

Download the [PopQA](https://github.com/AlexTMallen/adaptive-retrieval?tab=readme-ov-file#popqa) dataset from its repository <https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv>, and put the file `popQA.tsv` into folder `data/popqa`.

```bash
mkdir -p data/popqa
wget -P data/popqa https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv
```

For cwq:

Download the [cwq](https://www.tau-nlp.sites.tau.ac.il/compwebq) dataset from its repository <https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1>, and put the file `cwq_dev.json` into folder `data/cwq`.

#### Dynamic Retrieval
```bash
cd src
python main.py -c {your_json_path}
```
your_json_path specifies your retreival parameters. Here is an example:
```json
{
    "model_name_or_path": "/liuzyai04/thuir/tyc/LLM/Llama-3.2-1B-Instruct",
    "method": "token",
    "dataset": "2wikimultihopqa",
    "data_path": "../data/2wikimultihopqa",
    "fewshot": 6,
    "sample": 300,
    "shuffle": false,
    "generate_max_length": 300,
    "query_formulation": "forward_all",
    "output_dir": "../result/llama3.2_1b_2wikimultihopqa",
    "retriever": "BM25",
    "es_index_name": "wiki",
    "retrieve_topk": 3,
    "hallucination_threshold": 0.1,
    "use_counter": true,
    "sentence_solver": "avg",
    "entity_solver": "avg"
}
```
The parameters that can be selected in the config file are as follows:

| parameter                 | meaning                                                      | example/options                                              |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `model_name_or_path`      | Hugging Face model.                                          | `meta-llama/Llama-2-13b-chat`                             |
| `method`                  | way to generate answers             | `non-retrieval`, `single-retrieval`, `token`, `fix-sentence-retrieval`, `fix-length-retrieval`, `dragin` |
| `dataset`                 | Dataset                                                      | `2wikimultihopqa`, `hotpotqa`, `cwq`, `popqa`          |
| `data_path`               | the folder where the data is located. If you use the above code to download the data, the folder will be `../data/dataset`. | `../data/2wikimultihopqa`                                    |
| `fewshot`                 | Few shot.                                                    | 6                                                            |
| `sample`                  | number of questions sampled from the dataset.<br />`-1` means use the entire data set. | 1000                                                         |
| `shuffle`                 | Whether to disrupt the data set.<br />Without this parameter, the data set will not be shuffled. | `true`, `false`(without)                                     |
| `generate_max_length`     | maximum generated length of a question                       | 64                                                           |
| `query_formulation`       | way to generate retrieval question.                          | main: `direct`, `real_words`<br />another options: `current_wo_wrong`, `current`, `forward_all`, `last_n_tokens`, `last_sentence` |
| `retrieve_keep_top_k`     | number of reserved tokens when generating a search question  | 35                                                           |
| `output_dir`              | The generated results will be stored in a folder with a numeric name at the output folder you gave. If the folder you give does not exist, one will be created. | `../result/2wikimultihopqa_llama2_13b`                       |
| `retriever`               | type of retriever.                                           | `BM25`, `SGPT`                                               |
| `retrieve_topk`           | number of related documents retained.                        | 3                                                            |
| `hallucination_threshold` | threshold at which a word is judged to be incorrect.         | 1.2                                                          |
| `check_real_words`        | Whether only content words participate in threshold judgment.<br />Without this parameter, all words will be considered. | `true`, `false`(without)                                     |
| `use_counter`             | Whether to use counters to count the number of generation, retrieval, number of problems, number of tokens generated, and number of sentences generated.<br />Without this parameter, the number will not be counted. | `true`, `false`(without)                                     |

If you are using BM25 as the retriever, you should also include the following parameters

| Parameter       | Meaning                                    | example |
| --------------- | ------------------------------------------ | ------- |
| `es_index_name` | The name of the index in the Elasticsearch | `wiki`  |




#### Data Augmentation:

```bash
python src/augment.py \
    --model_name llama3.2_1b \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300  \
    --topk 3
```

| **Parameter** | **Example/Options** |
| ------------------------------ | ---------------------------------------------------- |
| `model_name` | `llama3.2_1b`, `qwen2.5_1.5b`, `llama3_8b` |
| `dataset` | `2wikimultihopqa`, `hotpotqa`, `popqa`, `cwq` |
| `data_path` | folder to the saved data, such as `data/2wikimultihopqa` |
| `sample` | Number of questions to run |
| `topk` | retrieval number |

The results of data augmentation will be stored in the file `data_aug/{dataset}/{data_type}.json`.

If you want to apply data augmentation to a new dataset, the default data format for the augmented data is JSON. Each element in the array should include both a 'question' and an 'answer,' as shown in the example below.

```json
[
    {
        "question": "string",
        "answer": "string or list[string]",
    }
]
```

At this point, the input parameter `dataset` refers to the name of the dataset you’ve set, and `data_path` is the path to the JSON file mentioned above. The last filename in `data_path` will be treated as the `data_type`. The output file will be saved in `data_aug/{your_dataset_name}/{data_type}.json`.





#### Building Experts

By calling the `src/encode.py` file, you will generate a parameterized representation of the documents (Expert) for the given dataset. The parameters for this file are as follows:

| **Parameter**                  | **Example/Options**                                  |
| ------------------------------ | ---------------------------------------------------- |
| `model_name`                   | `llama3.2_1b`, `qwen2.5_1.5b`, `llama3_8b` |
| `dataset`                      | `2wikimultihopqa`, `hotpotqa`, `popqa`, `cwq` |
| `data_type`                    | Not set means using the entire dataset, otherwise, specify a particular data type |
| `with_cot`                     | If included, generate a CoT |
| `sample`                        | Number of questions to run |
| `augment_model`                | Model used for data augmentation. If not set, the current model will be used for augmentation |
| `per_device_train_batch_size`, `num_train_epochs`, `learning_rate` | Training parameters |
| `lora_rank`, `lora_alpha`       | LoRA parameters, dropout will be set to 0 |

When running for the first time with a specific LoRA parameter, an initial random parameter, `base_weight` will be created. All subsequent training will start from this base_weight.

All generated experts are stored in the `offline` folder. 
The specific location of the expert files is as follows:

```plain
offline/
├── {model_name}/
│   └── rank={lora_rank}_alpha={lora_alpha}/
│       ├── base_weight/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}/
│               └── aug_model={augment_model}/
│                   └── {data_type}/
│                       └── data_{did}/
│                           └── passage_{pid}/
|                               └── parameters
```

#### Generate

By calling the `src/inference.py` file, you will generate an "expert" of the documents for the given dataset. The parameters for this file are as follows:

| **Parameter**                  | **Example/Options**                                  |
| ------------------------------ | ---------------------------------------------------- |
| `model_name`                   | `llama3.2_1b`, `qwen2.5_1.5b`, `llama3_8b` |
| `dataset`                      | `2wikimultihopqa`, `hotpotqa`, `popqa`, `cwq` |
| `data_type`                    | Not set means using the entire dataset, otherwise, specify a particular data type |
| `with_cot`                     | If included, generate a CoT |
| `sample`                        | Number of questions to run |
| `augment_model`                | Model used for data augmentation. If not set, the current model will be used for augmentation |
| `per_device_train_batch_size`, `num_train_epochs`, `learning_rate` | Training parameters |
| `lora_rank`, `lora_alpha`       | LoRA parameters, dropout will be set to 0 |
| `max_new_tokens` | Number of generate tokens |
| `inference_method` | default set to 'combine' |

All generated results are stored in the `output` folder. The specific location of the parameter files is as follows:

```plain
offline/
├── {model_name}/
│   └── rank={lora_rank}_alpha={lora_alpha}/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}/
│               └── aug_model={augment_model}/
│                   └── {inference_method}/
│                       └── {data_type}/
│                           ├── config.json
│                           ├── predict.json
│                           └── result.txt
```
