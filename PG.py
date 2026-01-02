from os import read
from fastapi import BackgroundTasks
from transformers import AutoModelForCausalLM, AutoTokenizer,Trainer, TrainingArguments, AdamW,\
get_linear_schedule_with_warmup,DataCollatorWithPadding, Adafactor
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from hqq.models.hf.base import AutoHQQHFModel
from hqq.models.hf.mixtral import MixtralHQQ #Mixtral
from hqq.core.quantize import *
from hqq.models.hf.base import AutoHQQHFModel
import gym
import torch
from torch import nn
from torch.distributions import Categorical
import pandas as pd
import math
import torch.nn.functional as F

'''这个版本 动作，是每个expert + 还是 - 还是 0 ，方法是 PG'''

torch.autograd.set_detect_anomaly(True)

'''1. for model and data'''

def encode(examples): # 定义数据加载器
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

def collate_fn(batch): # 转换为 PyTorch Tensor
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    return {"input_ids": input_ids}

def data_pre(): # data prepare
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1") # 加载Wikitext-2数据集
    encoded_validation_dataset = dataset['validation'].map(encode, batched=True) # 只对验证集合进行 map 操作
    encoded_validation_dataset.set_format(type='python', columns=['input_ids'])
    test_dataset = encoded_validation_dataset.shuffle(seed=42).select(range(200))
    test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False, collate_fn=collate_fn)
    return test_loader

def eval_m(model, test_loader):
    model.eval()
    losses = []
    # progress_bar = tqdm(test_loader, desc="Evaluation")
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['input_ids'].to(device), batch['input_ids'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    return perplexity

def replace_linear_layers(model, quan_set):
    # print("# Start quantize model ")
    for i_layer in range(10):
        for j_expe in range(4):
            if quan_set[i_layer][j_expe] != 16: 
                model.model.layers[i_layer].block_sparse_moe.experts[j_expe].w1 = HQQLinear(model.model.layers[i_layer].block_sparse_moe.experts[j_expe].w1 , 
                                quant_config=BaseQuantizeConfig(nbits=quan_set[i_layer][j_expe], group_size=64), #quantization configuration
                                compute_dtype=torch.bfloat16, device='cuda', del_orig=False)
                
                model.model.layers[i_layer].block_sparse_moe.experts[j_expe].w2 = HQQLinear(model.model.layers[i_layer].block_sparse_moe.experts[j_expe].w2 , 
                                quant_config=BaseQuantizeConfig(nbits=quan_set[i_layer][j_expe], group_size=64), #quantization configuration
                                compute_dtype=torch.bfloat16, device='cuda', del_orig=False)
                
                model.model.layers[i_layer].block_sparse_moe.experts[j_expe].w3 = HQQLinear(model.model.layers[i_layer].block_sparse_moe.experts[j_expe].w3 ,
                                quant_config=BaseQuantizeConfig(nbits=quan_set[i_layer][j_expe], group_size=64), #quantization configuration
                                compute_dtype=torch.bfloat16, device='cuda', del_orig=False)
                
    # print("# End quanize model")
    return model 

# 处理函数
def state_map(A, B, C):
    result = A.copy()
    mask_decrease = (B == -1) & ((A == 8) | (A == 16))
    mask_increase = (B == 1) & (A == 8)
    
    result[mask_decrease & (A == 8)] = C[C < 8][-1]
    result[mask_decrease & (A == 16)] = C[C < 16][-1]
    result[mask_increase] = C[C > 8][0]
    
    return result.astype(int)  

class ExpertBitAllocationEnv(gym.Env):
    def __init__(self, num_layers=10, num_experts_per_layer=4, max_bits=240, max_steps=5):
        super(ExpertBitAllocationEnv, self).__init__()
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.num_experts = num_layers * num_experts_per_layer
        self.max_steps = max_steps
        self.max_bits = max_bits
        self.bits_options = np.array([-1, 0, 1])
        self.bit_real = np.array([1, 2, 3, 4, 8, 16])
        self.action_space = gym.spaces.MultiDiscrete([len(self.bits_options)] * self.num_experts)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.num_layers, self.num_experts_per_layer), dtype=np.float32)
        self.current_step = 0
        self.reset()
        self.trade = 10

    def reset(self):
        # self.state = np.full((self.num_layers, self.num_experts_per_layer), 8.0)
        self.state = [ [8, 4, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8],
                    [4, 8, 4, 8]]
        self.current_step = 0
        return np.array(self.state)

    def step(self, action):
        action_bits = self.bits_options[action]
        result = state_map(np.array(self.state), action_bits, self.bit_real)
        self.state = np.array(result)
        total_bits_used = np.sum(result)
        reward_bits = math.log(max(total_bits_used, 1e-10), 10)
        reward_ppi = self._get_reward(result)

        if total_bits_used > self.max_bits+5:
            reward = 0.1  # Penalize if over budget
        else:
            reward = reward_ppi
            

        # reward = 10000 ** reward_ppi / 10 - reward_bits

        self.current_step += 1
        done = self.current_step >= self.max_steps
        return np.array(self.state), reward, done, {}

    def _get_reward(self, result):
        quan_set = result
        quantized_model = copy.deepcopy(original_model).to(device)
        new_model = replace_linear_layers(quantized_model, quan_set)
        new_model = new_model.to(device)
        perplexity = eval_m(new_model, test_loader)
        reward = 1 / perplexity
        torch.cuda.empty_cache()
        return reward
    
class PolicyNetwork(nn.Module):
    def __init__(self, num_layers, num_experts_per_layer):
        super(PolicyNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.fc1 = nn.Linear(num_layers * num_experts_per_layer, 512)
        self.fc2 = nn.Linear(512, 256)
        self.action_head = nn.Linear(256, num_layers * num_experts_per_layer * 3)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_head(x)
        action_logits = action_logits.view(-1, self.num_layers, self.num_experts_per_layer, 3)
        return action_logits

def train_policy(env, num_episodes=200, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01):
    policy_net = PolicyNetwork(env.num_layers, env.num_experts_per_layer)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    Loss = []
    Reward = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        episode_loss = []
        loss_in = []
        reward_in = []
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                action = np.array(action).reshape(env.num_layers, env.num_experts_per_layer)
                action_logits = torch.ones(1, env.num_layers, env.num_experts_per_layer, 3, requires_grad=True) / 3.0
                action_probs = torch.softmax(action_logits, dim=-1)
            else:
                action_logits = policy_net(state)
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).squeeze()


            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            optimizer.zero_grad()

            flat_action_probs = action_probs.view(-1) # 1*10*4*3
            action_indices = np.unravel_index(action, (env.num_layers, env.num_experts_per_layer, 3))

            # print("action_indices",action_indices)
            action_indices = []
            for i in range(env.num_layers):
                for j in range(env.num_experts_per_layer):
                    action_index = np.ravel_multi_index((0, i, j, action[i, j]), (1, env.num_layers, env.num_experts_per_layer, 3))
                    action_indices.append(action_index)
            
            action_indices = torch.tensor(action_indices, dtype=torch.long)


            loss = -torch.log(flat_action_probs[action_indices]).sum() * reward
            loss.backward()
            optimizer.step()
            state = next_state

            loss_in.append(loss.detach().numpy())
            reward_in.append(reward)

            print(f"reward: {reward}")

        Loss.append(sum(loss_in) / len(loss_in))
        avg_rew = sum(reward_in) / len(reward_in)
        Reward.append(avg_rew)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode == num_episodes-1:
            print("#action", state)
        

        print(f"Episode: {episode}, Loss: {sum(loss_in) / len(loss_in)}, Reward: {avg_rew}")


    series = pd.Series(Loss)
    series.to_csv('./loss-data.csv', index=False)

    return policy_net
        



if __name__ == "__main__":

    device = "cuda"
    model_path = "./finetuned_wikitext2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    original_model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16)
    test_loader = data_pre()

    env = ExpertBitAllocationEnv()
    dqn = train_policy(env)
    

