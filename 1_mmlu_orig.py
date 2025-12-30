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


def estimate_model_size_in_gb(model):
   # 使用 BytesIO 对象模拟文件保存
   buffer = io.BytesIO()
   torch.save(model.state_dict(), buffer)  # 保存模型的 state_dict 到内存
   size_in_bytes = buffer.getbuffer().nbytes  # 获取内存中保存的字节数
   size_in_gb = size_in_bytes / (1024 ** 3)  # 转换为 GB
   return size_in_gb

def replace_linear_layers(model, quan_set):
   print("# Start quantize model ")
   for i_layer in range(36):

        HQQLinear.set_backend(HQQBackend.ATEN)

        model.model.layers[i_layer].mlp.down_proj = HQQLinear(model.model.layers[i_layer].mlp.down_proj,
                            quant_config=BaseQuantizeConfig(nbits=quan_set, axis= 0, group_size=64), #quantization configuration
                            compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
        
        model.model.layers[i_layer].mlp.gate_proj = HQQLinear(model.model.layers[i_layer].mlp.gate_proj,
                            quant_config=BaseQuantizeConfig(nbits=quan_set, axis= 0, group_size=64), #quantization configuration
                            compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
        
        model.model.layers[i_layer].mlp.up_proj = HQQLinear(model.model.layers[i_layer].mlp.up_proj,
                            quant_config=BaseQuantizeConfig(nbits=quan_set, axis= 0, group_size=64), #quantization configuration
                            compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
        
        
        model.model.layers[i_layer].self_attn.k_proj = HQQLinear(model.model.layers[i_layer].self_attn.k_proj,
                            quant_config=BaseQuantizeConfig(nbits=quan_set, axis= 0, group_size=64), #quantization configuration
                            compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
        
        model.model.layers[i_layer].self_attn.o_proj = HQQLinear(model.model.layers[i_layer].self_attn.o_proj,
                            quant_config=BaseQuantizeConfig(nbits=quan_set, axis= 0, group_size=64), #quantization configuration
                            compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
        
        model.model.layers[i_layer].self_attn.q_proj = HQQLinear(model.model.layers[i_layer].self_attn.q_proj,
                            quant_config=BaseQuantizeConfig(nbits=quan_set, axis= 0, group_size=64), #quantization configuration
                            compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
        
        model.model.layers[i_layer].self_attn.v_proj = HQQLinear(model.model.layers[i_layer].self_attn.v_proj,
                            quant_config=BaseQuantizeConfig(nbits=quan_set, axis= 0, group_size=64), #quantization configuration
                            compute_dtype=torch.bfloat16, device='cuda', del_orig=True)
              
   print("# End quanize model")
   return model


def main(ckpt_dir: str, param_size: str, model_type: str):
  
   run_results = {}
   output_filename = 'run_results_%s_%sb.json' % (model_type, param_size)
  
   model, tokenizer = load(ckpt_dir, model_type)
   model_size_gb = estimate_model_size_in_gb(model)
   print(f"Estimated Model size: {model_size_gb:.2f} GB")

#    new_model = replace_linear_layers(model, 2)
#    AutoHQQHFModel.save_quantized(model, './qued_model')

    



#    new_model = model

#    model_size_gb = estimate_model_size_in_gb(new_model)
#    print(f"New Model size: {model_size_gb:.2f} GB")


   start_time = time.time()
   i_tas = 0
   for task in TASKS:
       print('Testing %s ...' % task)
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


    #    i_tas = i_tas +1
    #    if i_tas == 5:
    #        break


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



'''
cd /home/lt/anaconda3/envs/Quan/lib/python3.8/site-packages/hqq/kernels/



nsys profile --delay 30 -o non-quan-B1 python test.py

nsys profile --delay 30 -o quan-B1 python test.py

'''