import os
import math
import gc
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# HQQ
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
from hqq.core.quantize import HQQBackend

# -----------------------------
# Paper-aligned experiment setup (Evaluation / Alg.2)
# -----------------------------
DATASET_SPECS = {
    # Wiki [30]
    "wiki": ("wikitext", "wikitext-2-raw-v1", "test", None),
    # MNews [31] (MultiNews)
    "mnews": ("multi_news", None, "test", None),
    # Samsum [32]
    "samsum": ("samsum", None, "test", None),
    # Persona-Chat [33] (conv_ai_2 is PersonaChat)
    "persona": ("conv_ai_2", None, "test", None),
    # Yelp [34]
    "yelp": ("yelp_review_full", None, "test", None),
}
TASK_KEYS = ["wiki", "mnews", "samsum", "persona", "yelp"]

# Paper says HQQ supports Int8/4/3/2/1 and original BF16. (BF16 = 2 bytes, INT1 = 1/8 byte)
BIT_LEVELS = [16, 8, 4, 3, 2, 1]
BYTES_PER_WEIGHT = {16: 2.0, 8: 1.0, 4: 0.5, 3: 0.375, 2: 0.25, 1: 0.125}

# 5 servers, arithmetic progression between 0.25 and 0.75 bytes/expert (paper wording)
SERVER_CAP_BYTES_PER_WEIGHT = [0.25, 0.375, 0.5, 0.625, 0.75]


@dataclass
class ServerState:
    # per-layer per-expert bitwidth setting
    bits: List[List[int]]  # shape [n_layers][n_experts]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Mixtral-style MoE helpers (works with transformers/models/mixtral)
# Your model is loaded via AutoModelForCausalLM; it ends up using Mixtral implementation in your traceback.
# -----------------------------
def get_moe_meta(model) -> Tuple[int, int]:
    """Return (n_layers, n_experts) for Mixtral-style block_sparse_moe."""
    layers = model.model.layers
    n_layers = len(layers)
    # assume all layers have same number of experts
    n_experts = len(layers[0].block_sparse_moe.experts)
    return n_layers, n_experts


def get_expert_linears(model, layer_idx: int, expert_idx: int):
    """Return (w1, w2, w3) linear modules for one expert in one layer."""
    expert = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
    return expert.w1, expert.w2, expert.w3


def count_expert_params(model) -> List[List[int]]:
    """params per (layer, expert) for budgeting."""
    n_layers, n_experts = get_moe_meta(model)
    counts = [[0 for _ in range(n_experts)] for _ in range(n_layers)]
    for l in range(n_layers):
        for e in range(n_experts):
            w1, w2, w3 = get_expert_linears(model, l, e)
            counts[l][e] = w1.weight.numel() + w2.weight.numel() + w3.weight.numel()
    return counts


# -----------------------------
# Request sampling aligned with paper:
# - 50 requests per serving
# - from 5 datasets
# - counts follow Normal(10,5) then normalized to sum=50
# -----------------------------
def sample_task_counts(total_requests: int, mu: float = 10.0, sigma: float = 5.0) -> Dict[str, int]:
    raw = np.random.normal(mu, sigma, size=len(TASK_KEYS))
    raw = np.clip(raw, 0.0, None)
    if raw.sum() < 1e-6:
        raw = np.ones_like(raw)
    scaled = raw / raw.sum() * total_requests
    counts = np.floor(scaled).astype(int)

    # fix rounding to exactly total_requests
    diff = total_requests - int(counts.sum())
    order = np.argsort(-(scaled - counts))  # largest fractional parts first
    for i in range(abs(diff)):
        idx = order[i % len(order)]
        counts[idx] += 1 if diff > 0 else -1

    return {TASK_KEYS[i]: int(counts[i]) for i in range(len(TASK_KEYS))}


def load_requests_from_datasets(max_per_task: int = 2000) -> Dict[str, List[str]]:
    """Load a pool of text samples per task from HF datasets."""
    pools = {}
    for key, (ds_name, ds_config, split, _) in DATASET_SPECS.items():
        if ds_config is None:
            ds = load_dataset(ds_name, split=split)
        else:
            ds = load_dataset(ds_name, ds_config, split=split)

        # keep a limited pool to avoid memory blow-up
        ds = ds.select(range(min(max_per_task, len(ds))))

        texts = []
        if key == "wiki":
            # wikitext lines
            for ex in ds:
                t = ex.get("text", "").strip()
                if len(t) > 50:
                    texts.append(t)
        elif key == "mnews":
            # multi_news: document & summary fields
            for ex in ds:
                doc = ex.get("document", "")
                summ = ex.get("summary", "")
                t = f"Summarize the following news.\n\n{doc}\n\nSummary:"
                if len(doc) > 100:
                    texts.append(t)
        elif key == "samsum":
            # samsum: dialogue/summary
            for ex in ds:
                d = ex.get("dialogue", "")
                t = f"Summarize this dialogue:\n\n{d}\n\nSummary:"
                if len(d) > 50:
                    texts.append(t)
        elif key == "persona":
            # conv_ai_2: dialogue is structured; take "dialog"
            for ex in ds:
                dialog = ex.get("dialog", None)
                if dialog and isinstance(dialog, list) and len(dialog) > 0:
                    # flatten a few turns
                    turns = []
                    for turn in dialog[:6]:
                        if isinstance(turn, dict):
                            # keys vary; try common ones
                            text = turn.get("text", "") or turn.get("utterance", "") or ""
                        else:
                            text = str(turn)
                        if text:
                            turns.append(text)
                    if turns:
                        t = "Dialogue:\n" + "\n".join(turns) + "\n\nReply:"
                        texts.append(t)
        elif key == "yelp":
            for ex in ds:
                t = ex.get("text", "")
                if isinstance(t, str) and len(t) > 50:
                    texts.append("Review:\n" + t + "\n\nSentiment:")
        else:
            raise ValueError(key)

        if len(texts) < 10:
            raise RuntimeError(f"Not enough samples loaded for {key}. Check dataset availability.")
        pools[key] = texts

    return pools


def build_serving_requests(pools: Dict[str, List[str]], total_requests: int) -> List[Tuple[str, str]]:
    """Return list of (task_key, prompt_text) length total_requests."""
    counts = sample_task_counts(total_requests)
    reqs = []
    for k, c in counts.items():
        for _ in range(c):
            reqs.append((k, random.choice(pools[k])))
    random.shuffle(reqs)
    return reqs


# -----------------------------
# Nearest-server assignment (paper Alg.2 line 4)
# We'll simulate positions in a 2D plane.
# -----------------------------
def allocate_nearest_servers(n_servers: int, n_reqs: int, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    servers = rng.random((n_servers, 2))
    users = rng.random((n_reqs, 2))
    d2 = ((users[:, None, :] - servers[None, :, :]) ** 2).sum(-1)
    return d2.argmin(-1).tolist()


# -----------------------------
# Collect expert activation frequencies using router logits
# -----------------------------
@torch.no_grad()
def collect_activation_counts(
    model,
    tokenizer,
    prompts: List[str],
    max_seq_len: int,
    batch_size: int = 1,
    topk: int = 2,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Return counts: [n_layers, n_experts] activation counts (how often each expert selected in top-k).
    Uses forward_hook on gate(router) output: router_logits.
    """
    n_layers, n_experts = get_moe_meta(model)
    counts = torch.zeros((n_layers, n_experts), dtype=torch.long, device="cpu")

    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, inp, out):
            # out: router_logits, typically [tokens, n_experts] or [bs*seq, n_experts]
            router_logits = out
            # top-k experts per token
            top_idx = torch.topk(router_logits, k=topk, dim=-1).indices  # [..., topk]
            top_idx = top_idx.reshape(-1)  # flatten
            bc = torch.bincount(top_idx.to(torch.int64), minlength=n_experts).cpu()
            counts[layer_idx] += bc
        return hook

    # register hooks
    for l in range(n_layers):
        gate = model.model.layers[l].block_sparse_moe.gate
        hooks.append(gate.register_forward_hook(make_hook(l)))

    model.eval()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc)

    # remove hooks
    for h in hooks:
        h.remove()

    return counts


# -----------------------------
# Paper Alg.2-inspired heuristic to produce per-server bits under capacity
# - First step: start from BF16 (16)
# - Later steps: keep top quarter experts high, bottom 3/4 inherit previous bits
# - If exceed capacity: iteratively down-quantize lowest-activated experts to next lower bit level
# -----------------------------
def next_lower_bit(cur: int) -> int:
    idx = BIT_LEVELS.index(cur)
    return BIT_LEVELS[min(idx + 1, len(BIT_LEVELS) - 1)]


def estimate_bytes(bits: List[List[int]], param_counts: List[List[int]]) -> float:
    total = 0.0
    for l in range(len(bits)):
        for e in range(len(bits[0])):
            total += param_counts[l][e] * BYTES_PER_WEIGHT[bits[l][e]]
    return total


def make_initial_bits(n_layers: int, n_experts: int, bit: int = 16) -> List[List[int]]:
    return [[bit for _ in range(n_experts)] for _ in range(n_layers)]


def build_bits_for_server(
    prev_bits: List[List[int]],
    activation_counts: torch.Tensor,  # [n_layers, n_experts], cpu
    param_counts: List[List[int]],
    capacity_bytes_per_weight: float,
) -> List[List[int]]:
    n_layers, n_experts = activation_counts.shape
    total_params = sum(sum(row) for row in param_counts)
    budget_bytes = capacity_bytes_per_weight * total_params

    # start from prev
    bits = [row[:] for row in prev_bits]

    # keep top quarter high in bits; bottom 3/4 inherit previous (already true).
    # "high" here: set to BF16 (16). (paper keeps first quarter high in bits)
    for l in range(n_layers):
        act = activation_counts[l].numpy()
        order = np.argsort(-act)  # desc
        top_q = max(1, n_experts // 4)
        for idx in order[:top_q]:
            bits[l][idx] = 16

    # enforce capacity: down-quantize from lowest-activated experts
    cur_bytes = estimate_bytes(bits, param_counts)
    if cur_bytes <= budget_bytes:
        return bits

    for l in range(n_layers):
        act = activation_counts[l].numpy()
        order_low = np.argsort(act)  # asc
        ptr = 0
        while cur_bytes > budget_bytes and ptr < n_experts:
            e = int(order_low[ptr])
            ptr += 1
            if bits[l][e] == BIT_LEVELS[-1]:
                continue
            bits[l][e] = next_lower_bit(bits[l][e])
            cur_bytes = estimate_bytes(bits, param_counts)

    # if still exceed, do global sweep until met or cannot reduce
    while cur_bytes > budget_bytes:
        changed = False
        # pick (l,e) with smallest activation among those not at min
        best = None
        best_act = None
        for l in range(n_layers):
            for e in range(n_experts):
                if bits[l][e] == BIT_LEVELS[-1]:
                    continue
                a = int(activation_counts[l, e].item())
                if best is None or a < best_act:
                    best = (l, e)
                    best_act = a
        if best is None:
            break
        l, e = best
        bits[l][e] = next_lower_bit(bits[l][e])
        changed = True
        cur_bytes = estimate_bytes(bits, param_counts)
        if not changed:
            break

    return bits


# -----------------------------
# Quantization: single-GPU friendly
# Strategy:
# 1) keep full model on CPU to save VRAM
# 2) for each expert linear: move THAT linear to GPU, quantize with HQQ on GPU, move quantized module back to CPU
# This avoids OOM caused by HQQ temporary tensors when full model stays on GPU.
# -----------------------------
def quantize_one_linear_gpu_then_offload(linear_module, nbits: int) -> torch.nn.Module:
    if nbits == 16:
        return linear_module  # keep as-is

    # move this linear to cuda (weights only)
    linear_module = linear_module.to("cuda")

    # HQQ backend (ATEN is usually safest)
    HQQLinear.set_backend(HQQBackend.ATEN)

    q = HQQLinear(
        linear_module,
        quant_config=BaseQuantizeConfig(nbits=nbits, group_size=64),
        compute_dtype=torch.bfloat16,
        device="cuda",
        del_orig=True,   # frees original weight inside the wrapper
    )

    # offload quantized module to cpu to release VRAM for next one
    q = q.to("cpu")

    # aggressive cleanup
    del linear_module
    torch.cuda.empty_cache()
    gc.collect()
    return q


def apply_expert_quantization_inplace_cpu_model(model_cpu, bits: List[List[int]]):
    n_layers, n_experts = get_moe_meta(model_cpu)
    assert len(bits) == n_layers and len(bits[0]) == n_experts

    # quantify only experts (w1/w2/w3); keep attention/embeddings untouched
    for l in range(n_layers):
        for e in range(n_experts):
            b = bits[l][e]
            if b == 16:
                continue
            w1, w2, w3 = get_expert_linears(model_cpu, l, e)
            model_cpu.model.layers[l].block_sparse_moe.experts[e].w1 = quantize_one_linear_gpu_then_offload(w1, b)
            model_cpu.model.layers[l].block_sparse_moe.experts[e].w2 = quantize_one_linear_gpu_then_offload(w2, b)
            model_cpu.model.layers[l].block_sparse_moe.experts[e].w3 = quantize_one_linear_gpu_then_offload(w3, b)

    return model_cpu


# -----------------------------
# Main simulation (10 servings, 50 req each, 5 servers)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--servings", type=int, default=10)
    ap.add_argument("--requests_per_serving", type=int, default=50)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--act_batch_size", type=int, default=1)
    ap.add_argument("--save_bits_json", type=str, default="server_bits_plan.json")
    ap.add_argument("--do_quantize", action="store_true", help="If set, actually quantize models for each server and save.")
    ap.add_argument("--save_dir", type=str, default="./quantized_servers")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    tokenizer.padding_side = "left"

    # Load base model:
    # - keep on CPU by default to save VRAM
    # - we'll move to GPU only when collecting activations
    model_cpu = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model_cpu.eval()

    n_layers, n_experts = get_moe_meta(model_cpu)
    param_counts = count_expert_params(model_cpu)

    # init per-server bits (start BF16)
    servers = [ServerState(bits=make_initial_bits(n_layers, n_experts, 16)) for _ in range(5)]

    # load request pools
    pools = load_requests_from_datasets(max_per_task=2000)

    plan_log = {
        "server_cap_bytes_per_weight": SERVER_CAP_BYTES_PER_WEIGHT,
        "servings": [],
    }

    for s in range(args.servings):
        # sample requests
        reqs = build_serving_requests(pools, args.requests_per_serving)
        # allocate to nearest server
        assign = allocate_nearest_servers(n_servers=5, n_reqs=len(reqs), seed=args.seed + s)

        # group prompts per server
        per_server_prompts = [[] for _ in range(5)]
        for idx, (task_key, prompt) in enumerate(reqs):
            per_server_prompts[assign[idx]].append(prompt)

        serving_entry = {"serving_idx": s, "servers": []}

        # Collect activations server-by-server using original model on GPU
        model_gpu = model_cpu.to("cuda")  # temporary
        for i in range(5):
            prompts_i = per_server_prompts[i]
            if len(prompts_i) == 0:
                # no requests => keep previous bits
                serving_entry["servers"].append({
                    "server_id": i,
                    "n_requests": 0,
                    "bytes_per_weight_cap": SERVER_CAP_BYTES_PER_WEIGHT[i],
                })
                continue

            act_counts = collect_activation_counts(
                model_gpu, tokenizer, prompts_i,
                max_seq_len=args.max_seq_len,
                batch_size=args.act_batch_size,
                topk=2,
                device="cuda",
            )

            # build bits according to heuristic / Alg.2
            new_bits = build_bits_for_server(
                prev_bits=servers[i].bits,
                activation_counts=act_counts,
                param_counts=param_counts,
                capacity_bytes_per_weight=SERVER_CAP_BYTES_PER_WEIGHT[i],
            )
            servers[i].bits = new_bits

            serving_entry["servers"].append({
                "server_id": i,
                "n_requests": len(prompts_i),
                "bytes_per_weight_cap": SERVER_CAP_BYTES_PER_WEIGHT[i],
                "est_total_bytes": estimate_bytes(new_bits, param_counts),
            })

        # move back to CPU to release VRAM
        model_cpu = model_gpu.to("cpu")
        del model_gpu
        torch.cuda.empty_cache()
        gc.collect()

        plan_log["servings"].append(serving_entry)
        print(f"[Serving {s}] planned bits updated.")

    # save plan
    with open(args.save_bits_json, "w", encoding="utf-8") as f:
        json.dump(
            {"n_layers": n_layers, "n_experts": n_experts, "plan_log": plan_log, "final_bits": [sv.bits for sv in servers]},
            f, ensure_ascii=False, indent=2
        )
    print(f"Saved bit plan to {args.save_bits_json}")

    # Optional: actually quantize and save 5 server models (sequentially to fit single GPU)
    if args.do_quantize:
        os.makedirs(args.save_dir, exist_ok=True)
        for i in range(5):
            print(f"[Quantize] Server {i} ...")
            model_cpu = AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
            )
            model_cpu.eval()

            model_cpu = apply_expert_quantization_inplace_cpu_model(model_cpu, servers[i].bits)

            out_dir = os.path.join(args.save_dir, f"server_{i}")
            os.makedirs(out_dir, exist_ok=True)
            model_cpu.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            del model_cpu
            gc.collect()
            print(f"[Quantize] Server {i} saved to {out_dir}")


if __name__ == "__main__":
    main()

