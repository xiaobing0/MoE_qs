

import os
import gc
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# HQQ (optional for reward_mode=ppl)
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig, HQQBackend


# -----------------------------
# Paper-aligned experiment settings
# -----------------------------
TASK_KEYS = ["wiki", "mnews", "samsum", "persona", "yelp"]  # :contentReference[oaicite:5]{index=5}
N_TASKS = 5
N_SERVERS = 5

# 0.25 ~ 0.75 bytes per weight (paper says bytes per expert; we use "per weight" budget like earlier code)
SERVER_CAP_BYTES_PER_WEIGHT = [0.25, 0.375, 0.5, 0.625, 0.75]  # arithmetic progression :contentReference[oaicite:6]{index=6}

# Simplified "next bit selection": choose among BF16/Int8/Int4 (HQQ supports many; this is runnable)
BITS = [16, 8, 4]
BITS_TO_IDX = {b: i for i, b in enumerate(BITS)}
BYTES_PER_WEIGHT = {16: 2.0, 8: 1.0, 4: 0.5}  # BF16 two bytes :contentReference[oaicite:7]{index=7}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Env time-slot setup: "ten times servings, each time contains 50 requests ... N(10,5)" :contentReference[oaicite:8]{index=8}
DEFAULT_REQUESTS_PER_SLOT = 50

# Reward weights (α1, α2) exist in paper :contentReference[oaicite:9]{index=9}
DEFAULT_ALPHA1 = 0.0001
DEFAULT_ALPHA2 = 0.01


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_task_counts(total_requests: int) -> np.ndarray:
    """
    Paper: samples consistent with normal distribution X ~ N(10,5) for five datasets :contentReference[oaicite:10]{index=10}
    We'll sample 5 numbers from N(10,5), clip to >=0, then normalize to sum=total_requests.
    """
    raw = np.random.normal(loc=10.0, scale=5.0, size=N_TASKS)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() < 1e-6:
        raw = np.ones_like(raw)
    scaled = raw / raw.sum() * total_requests
    counts = np.floor(scaled).astype(int)
    diff = total_requests - int(counts.sum())
    frac = scaled - counts
    order = np.argsort(-frac)
    for i in range(abs(diff)):
        counts[order[i % N_TASKS]] += 1 if diff > 0 else -1
    return counts


def split_requests_by_probs(task_counts: np.ndarray, probs_task_to_server: np.ndarray) -> np.ndarray:
    """
    Given per-task request counts and per-task distribution over servers,
    return assigned_counts[server, task].
    """
    assigned = np.zeros((N_SERVERS, N_TASKS), dtype=np.int32)
    for t in range(N_TASKS):
        p = probs_task_to_server[t]
        p = p / (p.sum() + 1e-9)
        # Multinomial split
        assigned[:, t] = np.random.multinomial(int(task_counts[t]), p)
    return assigned


# -----------------------------
# Optional: small loader for PPL evaluation (reward_mode=ppl)
# -----------------------------
def build_wikitext_loader(tokenizer, n_samples=64, seq_len=256, batch_size=8):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")["validation"]
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    def encode(ex):
        out = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=seq_len)
        return {"input_ids": out["input_ids"]}

    ds = ds.map(encode, remove_columns=ds.column_names)
    ds.set_format(type="python", columns=["input_ids"])

    def collate(batch):
        ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        return {"input_ids": ids}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


@torch.no_grad()
def eval_ppl(model, loader) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["input_ids"].to(DEVICE)
        out = model(input_ids=x, labels=x)
        losses.append(float(out.loss))
    return float(np.exp(np.mean(losses)))


# -----------------------------
# Model param counting for memory budget
# -----------------------------
def get_expert_param_counts(model, num_layers: int, num_experts: int) -> np.ndarray:
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    for l in range(num_layers):
        for e in range(num_experts):
            ex = model.model.layers[l].block_sparse_moe.experts[e]
            counts[l, e] = ex.w1.weight.numel() + ex.w2.weight.numel() + ex.w3.weight.numel()
    return counts


def estimate_server_bytes(bits_mat: np.ndarray, param_counts: np.ndarray) -> float:
    # bits_mat: [L, M]
    total = 0.0
    for l in range(bits_mat.shape[0]):
        for e in range(bits_mat.shape[1]):
            b = int(bits_mat[l, e])
            total += float(param_counts[l, e]) * (b / 8.0)
    return total


def max_server_budget_bytes(cap_bytes_per_weight: float, param_counts: np.ndarray) -> float:
    # capacity is expressed as bytes per weight (paper phrase "bytes per expert"; we implement per-weight budget)
    total_params = float(param_counts.sum())
    return cap_bytes_per_weight * total_params


# -----------------------------
# Switching cost C_i
# -----------------------------
def switching_cost(prev_bits: np.ndarray, new_bits: np.ndarray) -> float:
    # simple: count changed experts (or L1 diff normalized)
    changed = (prev_bits != new_bits).sum()
    return float(changed)


# -----------------------------
# Latency l_i (proxy): backlog / service_rate
# -----------------------------
def service_rate_from_bits(bits_mat: np.ndarray) -> float:
    # average bit -> higher throughput (proxy)
    avg_bit = float(bits_mat.mean())
    # normalize so BF16(16) ~ 1.0, Int8 ~ 1.3, Int4 ~ 1.6 (proxy, tunable)
    if avg_bit >= 12:
        return 1.0
    if avg_bit >= 6:
        return 1.3
    return 1.6


# -----------------------------
# Performance p_i (proxy): degrade when bits lower
# We average reward per task to eliminate impact of amount :contentReference[oaicite:11]{index=11}
# -----------------------------
def proxy_task_quality(avg_bit: float) -> float:
    # Map bit -> "quality score" (higher better), roughly inverse of ppl
    # BF16 best, int8 medium, int4 worse
    # bounded (0,1]
    if avg_bit >= 12:
        return 1.0
    if avg_bit >= 6:
        return 0.92
    return 0.82


# -----------------------------
# Quantization for reward_mode=ppl (VERY SLOW)
# -----------------------------
def hqq_quantize_experts_inplace(model, bits_mat: np.ndarray, num_layers: int, num_experts: int):
    HQQLinear.set_backend(HQQBackend.ATEN)
    for l in range(num_layers):
        for e in range(num_experts):
            b = int(bits_mat[l, e])
            if b == 16:
                continue
            ex = model.model.layers[l].block_sparse_moe.experts[e]
            ex.w1 = HQQLinear(ex.w1, quant_config=BaseQuantizeConfig(nbits=b, group_size=64),
                              compute_dtype=torch.bfloat16, device=DEVICE, del_orig=True)
            ex.w2 = HQQLinear(ex.w2, quant_config=BaseQuantizeConfig(nbits=b, group_size=64),
                              compute_dtype=torch.bfloat16, device=DEVICE, del_orig=True)
            ex.w3 = HQQLinear(ex.w3, quant_config=BaseQuantizeConfig(nbits=b, group_size=64),
                              compute_dtype=torch.bfloat16, device=DEVICE, del_orig=True)


# -----------------------------
# Environment (document-defined MDP)
# st = ({B_i^t}, K^t) and at = ({B_i^{t+1}}, κ:K_t→E) :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}
# rt = Σ_i {p_i - α1*l_i - α2*C_i}, rt=-1 if exceeds memory limitation :contentReference[oaicite:14]{index=14}
# -----------------------------
@dataclass
class EnvConfig:
    num_layers: int
    num_experts: int
    max_steps: int
    requests_per_slot: int
    alpha1: float
    alpha2: float
    reward_mode: str  # "proxy" or "ppl"


class MoEEdgeEnv:
    def __init__(self, model_ref, tokenizer, param_counts: np.ndarray, cfg: EnvConfig, ppl_loader=None):
        self.model_ref = model_ref  # base model (for ppl mode cloning)
        self.tokenizer = tokenizer
        self.param_counts = param_counts
        self.cfg = cfg
        self.ppl_loader = ppl_loader

        self.step_idx = 0
        self.bits = None  # [server, L, M]
        self.queue = None # [server, task] unfinished requests
        self.reset()

    def reset(self):
        self.step_idx = 0
        # init bits: BF16 everywhere
        self.bits = np.full((N_SERVERS, self.cfg.num_layers, self.cfg.num_experts), 16, dtype=np.int32)
        # init queues empty
        self.queue = np.zeros((N_SERVERS, N_TASKS), dtype=np.int32)
        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        # state = concat(bits normalized, queue normalized)
        bits_norm = (self.bits / 16.0).reshape(N_SERVERS, -1)                 # [S, L*M]
        queue_norm = (self.queue / max(1, self.cfg.requests_per_slot)).astype(np.float32)  # [S, T]
        st = np.concatenate([bits_norm, queue_norm], axis=1).astype(np.float32)            # [S, L*M+T]
        return torch.tensor(st, dtype=torch.float32, device=DEVICE)

    def step(self, new_bits: np.ndarray, probs_task_to_server: np.ndarray):
        """
        new_bits: [server, L, M] in {16,8,4}
        probs_task_to_server: [task, server] row-softmax
        """
        assert new_bits.shape == self.bits.shape
        assert probs_task_to_server.shape == (N_TASKS, N_SERVERS)

        # memory constraint per server
        for i in range(N_SERVERS):
            used = estimate_server_bytes(new_bits[i], self.param_counts)
            budget = max_server_budget_bytes(SERVER_CAP_BYTES_PER_WEIGHT[i], self.param_counts)
            if used > budget:
                # rt=-1 if exceeds memory limitation :contentReference[oaicite:15]{index=15}
                self.step_idx += 1
                done = (self.step_idx >= self.cfg.max_steps)
                self.bits = new_bits
                # queues still evolve (optional), but reward fixed -1 by paper
                return self._get_state(), -1.0, done, {"memory_violation": True}

        # sample new incoming requests (per task) and assign
        incoming = sample_task_counts(self.cfg.requests_per_slot)
        assigned = split_requests_by_probs(incoming, probs_task_to_server)  # [server, task]

        # update queues: add incoming
        self.queue = self.queue + assigned

        # compute per-server latency proxy and serve some requests
        latencies = []
        for i in range(N_SERVERS):
            rate = service_rate_from_bits(new_bits[i])
            # serve up to capacity proportional to rate
            serve_cap = int(round(rate * (self.cfg.requests_per_slot / N_SERVERS)))
            # simple serve: reduce queue in total
            total_q = int(self.queue[i].sum())
            served = min(total_q, serve_cap)
            if total_q > 0:
                # serve proportionally across tasks
                frac = self.queue[i] / (total_q + 1e-9)
                dec = np.floor(frac * served).astype(np.int32)
                # fix rounding
                diff = served - int(dec.sum())
                for k in np.argsort(-(frac - dec))[:max(0, diff)]:
                    dec[k] += 1
                self.queue[i] = np.maximum(self.queue[i] - dec, 0)
            # latency proxy: backlog / serve_cap
            lat = float(total_q) / float(max(1, serve_cap))
            latencies.append(lat)

        # switching cost
        switch_costs = [switching_cost(self.bits[i], new_bits[i]) for i in range(N_SERVERS)]

        # performance term p_i: average over tasks (paper says average to eliminate amount impact) :contentReference[oaicite:16]{index=16}
        # Here we compute per-server quality based on avg bit. Then average across tasks (proxy).
        perf_terms = []
        if self.cfg.reward_mode == "proxy":
            for i in range(N_SERVERS):
                q = proxy_task_quality(float(new_bits[i].mean()))
                perf_terms.append(q)
        elif self.cfg.reward_mode == "ppl":
            # VERY SLOW: clone base model, quantize, compute ppl -> quality=1/ppl
            perf_terms = []
            for i in range(N_SERVERS):
                m = copy.deepcopy(self.model_ref).to(DEVICE)
                hqq_quantize_experts_inplace(m, new_bits[i], self.cfg.num_layers, self.cfg.num_experts)
                ppl = eval_ppl(m, self.ppl_loader)
                perf_terms.append(1.0 / max(ppl, 1e-9))
                del m
                torch.cuda.empty_cache()
                gc.collect()
        else:
            raise ValueError(self.cfg.reward_mode)

        # reward as paper: sum_i {p_i - α1*l_i - α2*C_i} :contentReference[oaicite:17]{index=17}
        reward = 0.0
        for i in range(N_SERVERS):
            reward += perf_terms[i] - self.cfg.alpha1 * latencies[i] - self.cfg.alpha2 * switch_costs[i]

        # advance
        self.bits = new_bits
        self.step_idx += 1
        done = (self.step_idx >= self.cfg.max_steps)  # max interaction step Stp :contentReference[oaicite:18]{index=18}
        return self._get_state(), float(reward), done, {
            "incoming": incoming.tolist(),
            "latency": latencies,
            "switch_cost": switch_costs,
            "perf": perf_terms,
            "memory_violation": False
        }


# -----------------------------
# Policy Network (PG)
# Output:
# 1) bits logits per server per (layer,expert) over {16,8,4}
# 2) assignment logits per task over servers (κ approximation)
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int, num_layers: int, num_experts: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # bits: [S, L, M, 3]
        self.bits_head = nn.Linear(hidden, num_layers * num_experts * len(BITS))
        # assign: [T, S]
        self.assign_head = nn.Linear(hidden, N_TASKS * N_SERVERS)

    def forward(self, s: torch.Tensor):
        # s: [S, state_dim]
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))

        bits_logits = self.bits_head(x)  # [S, L*M*3]
        bits_logits = bits_logits.view(N_SERVERS, self.num_layers, self.num_experts, len(BITS))

        assign_logits = self.assign_head(x)  # [S, T*S] but we want global; simplest: use server-0 embedding
        # take server-0 as global controller feature (you can also mean-pool across servers)
        g = assign_logits[0].view(N_TASKS, N_SERVERS)  # [T, S]

        return bits_logits, g


def sample_action_from_policy(bits_logits: torch.Tensor, assign_logits: torch.Tensor):
    """
    bits_logits: [S,L,M,3]
    assign_logits: [T,S]
    """
    # bits: factorized categorical per expert
    bits_dist = torch.distributions.Categorical(logits=bits_logits)
    bits_idx = bits_dist.sample()  # [S,L,M] in {0,1,2}
    bits_logp = bits_dist.log_prob(bits_idx).sum()

    # assignment: categorical distribution per task across servers; sample a server id for each task
    assign_dist = torch.distributions.Categorical(logits=assign_logits)
    assign_server = assign_dist.sample()  # [T] each in {0..S-1}
    assign_logp = assign_dist.log_prob(assign_server).sum()

    # convert assignment into probabilities (for env multinomial split): use softmax probs
    assign_probs = torch.softmax(assign_logits, dim=-1)  # [T,S]

    return bits_idx, assign_probs, (bits_logp + assign_logp)


def bits_idx_to_bits(bits_idx: torch.Tensor) -> np.ndarray:
    # bits_idx: [S,L,M] in {0,1,2} -> bits {16,8,4}
    idx = bits_idx.detach().cpu().numpy()
    out = np.zeros_like(idx, dtype=np.int32)
    for k, b in enumerate(BITS):
        out[idx == k] = b
    return out


# -----------------------------
# REINFORCE training loop (no pretraining)
# -----------------------------
def train_pg(env: MoEEdgeEnv, policy: PolicyNet, episodes: int, lr: float, gamma: float):
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # simple baseline (moving average)
    baseline = 0.0
    baseline_beta = 0.9

    for ep in range(1, episodes + 1):
        s = env.reset()

        logps = []
        rewards = []
        infos = []

        done = False
        while not done:
            bits_logits, assign_logits = policy(s)
            bits_idx, assign_probs, logp = sample_action_from_policy(bits_logits, assign_logits)

            new_bits = bits_idx_to_bits(bits_idx)                  # [S,L,M] int
            probs_task_to_server = assign_probs.detach().cpu().numpy()  # [T,S] float

            s_next, r, done, info = env.step(new_bits, probs_task_to_server)

            logps.append(logp)
            rewards.append(r)
            infos.append(info)
            s = s_next

        # compute returns
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # update baseline
        ep_return = float(returns_t[0].item())
        baseline = baseline_beta * baseline + (1 - baseline_beta) * ep_return
        adv = returns_t - baseline

        # policy gradient loss
        logps_t = torch.stack(logps)
        loss = -(logps_t * adv.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            mem_viol = sum(1 for x in infos if x.get("memory_violation", False))
            avg_r = float(np.mean(rewards))
            print(f"[EP {ep:4d}] loss={float(loss):.4f} ep_return={ep_return:.3f} avg_step_reward={avg_r:.3f} mem_violation_steps={mem_viol}")

    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="HF path or local dir of Llama-MoE style model")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=5, help="Stp (max interaction steps per episode) :contentReference[oaicite:19]{index=19}")
    ap.add_argument("--requests_per_slot", type=int, default=DEFAULT_REQUESTS_PER_SLOT)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--alpha1", type=float, default=DEFAULT_ALPHA1)
    ap.add_argument("--alpha2", type=float, default=DEFAULT_ALPHA2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--reward_mode", type=str, default="proxy", choices=["proxy", "ppl"])
    ap.add_argument("--ppl_samples", type=int, default=64)
    ap.add_argument("--ppl_seq_len", type=int, default=256)
    ap.add_argument("--ppl_batch", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    tokenizer.padding_side = "left"

    # Keep reference model on CPU to reduce VRAM pressure; only used for ppl reward mode via deepcopy+to(DEVICE)
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model_ref.eval()

    # count params per expert for memory constraint
    param_counts = get_expert_param_counts(model_ref, args.layers, args.experts)

    ppl_loader = None
    if args.reward_mode == "ppl":
        ppl_loader = build_wikitext_loader(
            tokenizer,
            n_samples=args.ppl_samples,
            seq_len=args.ppl_seq_len,
            batch_size=args.ppl_batch
        )

    cfg = EnvConfig(
        num_layers=args.layers,
        num_experts=args.experts,
        max_steps=args.max_steps,
        requests_per_slot=args.requests_per_slot,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        reward_mode=args.reward_mode
    )

    env = MoEEdgeEnv(model_ref=model_ref, tokenizer=tokenizer, param_counts=param_counts, cfg=cfg, ppl_loader=ppl_loader)

    # state dim: per server: L*M bits + T queue
    state_dim = args.layers * args.experts + N_TASKS
    policy = PolicyNet(state_dim=state_dim, hidden=args.hidden, num_layers=args.layers, num_experts=args.experts).to(DEVICE)

    print(f"Training PG (no pretrain) | reward_mode={args.reward_mode} | episodes={args.episodes}")
    train_pg(env, policy, episodes=args.episodes, lr=args.lr, gamma=args.gamma)

    # Save policy
    os.makedirs("./ckpt_pg", exist_ok=True)
    torch.save(policy.state_dict(), "./ckpt_pg/policy.pt")
    print("Saved policy to ./ckpt_pg/policy.pt")


if __name__ == "__main__":
    main()

    

