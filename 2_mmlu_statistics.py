import argparse
import json
import os
import time


import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


from hqq.models.hf.base import AutoHQQHFModel
from hqq.models.hf.mixtral import MixtralHQQ #Mixtral
from hqq.core.quantize import *
from hqq.models.hf.base import AutoHQQHFModel
import io

choices = ["A", "B", "C", "D"]

TASKS = [
       'abstract_algebra',
       'anatomy',
       'astronomy',
       'business_ethics',
       'clinical_knowledge',
       'college_biology',
       'college_chemistry',
       'college_computer_science',
       'college_mathematics',
       'college_medicine',
       'college_physics',
       'computer_security',
       'conceptual_physics',
       'econometrics',
       'electrical_engineering',
       'elementary_mathematics',
       'formal_logic',
       'global_facts',
       'high_school_biology',
       'high_school_chemistry',
       'high_school_computer_science',
       'high_school_european_history',
       'high_school_geography',
       'high_school_government_and_politics',
       'high_school_macroeconomics',
       'high_school_mathematics',
       'high_school_microeconomics',
       'high_school_physics',
       'high_school_psychology',
       'high_school_statistics',
       'high_school_us_history',
       'high_school_world_history',
       'human_aging',
       'human_sexuality',
       'international_law',
       'jurisprudence',
       'logical_fallacies',
       'machine_learning',
       'management',
       'marketing',
       'medical_genetics',
       'miscellaneous',
       'moral_disputes',
       'moral_scenarios',
       'nutrition',
       'philosophy',
       'prehistory',
       'professional_accounting',
       'professional_law',
       'professional_medicine',
       'professional_psychology',
       'public_relations',
       'security_studies',
       'sociology',
       'us_foreign_policy',
       'virology',
       'world_religions']


def compute_metric(output_filename):
   with open(output_filename, 'r') as f:
       run_results = json.load(f)
   total_acc = 0
   total_num = 0
   for task in run_results:
       acc = 0
       pred_answers = run_results[task]['pred_answers']
       gold_answers = run_results[task]['gold_answers']
       for pred, gold in zip(pred_answers, gold_answers):
           if pred == gold: acc += 1
       print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
       total_acc += acc
       total_num += len(gold_answers)
   print("ACC-all: %.4f" % (total_acc/total_num))


def format_subject(subject):
   l = subject.split("_")
   s = ""
   for entry in l:
       s += " " + entry
   return s


def format_example(df, idx, include_answer=True):
   prompt = df.iloc[idx, 0]
   k = df.shape[1] - 2
   for j in range(k):
       prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
   prompt += "\nAnswer:"
   if include_answer:
       prompt += " {}\n\n".format(df.iloc[idx, k + 1])
   return prompt


def gen_prompt(train_df, subject, k=-1):
   prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
   if k == -1:
       k = train_df.shape[0]
   for i in range(k):
       prompt += format_example(train_df, i)
   return prompt


def prepare_input(tokenizer, prompts):
   input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
   input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
   for t in input_tokens:
       if torch.is_tensor(input_tokens[t]):
           input_tokens[t] = input_tokens[t].to('cuda')
   return input_tokens


def calculate_proportions(tensor):
    total_sum = tensor.sum().item()
    proportions = tensor / total_sum
    return proportions.tolist()


def load(ckpt_dir, model_type):
   n_gpus = torch.cuda.device_count()


   if model_type == 'llama':
       # we use tensor parallel for loading llama
       tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")      
       model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16)
       model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
       tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
       tokenizer.bos_token_id = 1
   else:
       # mpt-30b's tokenizer only has the fast version
       use_fast = "mosaicml/mpt-30b" in ckpt_dir
       # however, tensor parallel for running falcon will occur bugs
       tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=use_fast, padding_side="left",trust_remote_code=True)
       model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.bfloat16, trust_remote_code=True)
       if tokenizer.pad_token_id is None:
           if tokenizer.eos_token_id is not None:
               tokenizer.pad_token_id = tokenizer.eos_token_id
           else:
               tokenizer.pad_token_id = 0
   model.eval()
   return model, tokenizer


def batch_split(prompts, batch_num):
   batch_prompts = []
   mini_batch = []
   for prompt in prompts:
       mini_batch.append(prompt)
       if len(mini_batch) == batch_num:
           batch_prompts.append(mini_batch)
           mini_batch = []
   if len(mini_batch) != 0:
       batch_prompts.append(mini_batch)
   return batch_prompts


def batch_infer(model, tokenizer, prompts):
   batch_size = 1
   answers = []
   for batch_input in tqdm(batch_split(prompts, batch_size)):
       encode_inputs = prepare_input(tokenizer, batch_input)
       outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
       answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
   answers = [answer[-1] for answer in answers]
   return answers

def main(ckpt_dir: str, param_size: str, model_type: str):
   
   device = "cuda" 
   run_results = {}
   output_filename = 'run_results_%s_%sb.json' % (model_type, param_size)
   model, tokenizer = load(ckpt_dir, model_type)

   print(next(model.parameters()).device)

   # '''2. 添加专家激活数目统计的hook'''
   expert_activations = {i: torch.zeros(4, device="cuda", dtype=torch.float32) for i in range(24)}

   def moe_forward_hook(layer_idx):
        def hook(module, inputs, output):
            # output: router logits, shape usually [bs, seq, num_experts] or [tokens, num_experts]
            topk = 2
            idx = output.topk(topk, dim=-1).indices.reshape(-1).to(torch.long)
            expert_activations[layer_idx].index_add_(0, idx, torch.ones(idx.numel(), device=idx.device, dtype=expert_activations[layer_idx].dtype))
        return hook
  
    # 为每个需要的层注册钩子
   for i  in range(24):
        # if i == 23:  # 只为第23层添加，但可以扩展到其他层
        model.model.layers[i].block_sparse_moe.gate.register_forward_hook(moe_forward_hook(i))
        
   print(model)


#    model.to(device)




   start_time = time.time()
   i_tas = 0
   Final_save = []
   s_ind = 0
   ss_ind = 0
   for task in TASKS:
       print('Testing %s ...' % task)
       Final_save.append([])
       records = []
       dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
       test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
       for i in range(test_df.shape[0]):
           # get prompt and make sure it fits
           k = args.ntrain
           prompt_end = format_example(test_df, i, include_answer=False)
           train_prompt = gen_prompt(dev_df, task, k)
           prompt = train_prompt + prompt_end
           while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
               prompt_split = prompt.split("\n\n")
               prompt_split.pop(1)
               prompt = '\n\n'.join(prompt_split)
           label = test_df.iloc[i, test_df.shape[1]-1]
           records.append({'prompt':prompt, 'answer':label})


       pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
       gold_answers = [record['answer'] for record in records]
       run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}

       '''3. 输出所有token在4个专家上激活情况'''
       print("### LT_Expert activations:", expert_activations) #LT
       for layer_idx, activations in expert_activations.items():
           Final_save[ss_ind].append(calculate_proportions(activations.cpu()))            


       i_tas = i_tas +1
       if i_tas == 2:
           break


    #    break


   with open(output_filename, 'w') as f:
       json.dump(run_results, f, ensure_ascii=False, indent=2)
  
   compute_metric(output_filename)
   end_time = time.time()
   print("total run time %.2f" % (end_time - start_time))
  
  


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
#    parser.add_argument('--ckpt_dir', type=str, default="Isotonic/TinyQwex-4x620M-MoE")
   parser.add_argument('--ckpt_dir', type=str, default="chestnutlzj/MoE-Qwen-4x1.8B-pretrain-50000-ckpt")
#    parser.add_argument('--ckpt_dir', type=str, default="Qwen/Qwen2.5-3B-Instruct")
   parser.add_argument('--param_size', type=str)
   parser.add_argument('--model_type', type=str, default='llamamoe')
   parser.add_argument('--data_dir', type=str, default='./data/')
   parser.add_argument('--ntrain', type=int, default=5)
   args = parser.parse_args()
   main(args.ckpt_dir, args.param_size, args.model_type)