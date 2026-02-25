import argparse
import inspect
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from importlib.metadata import version
from tqdm import tqdm

from model import DreamModel

from lib.prune_dream import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, prune_sink, prune_sink_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot


def forward_process(batch, mask_id, sampling_eps=1e-3):
    """Dream-style forward process with uniform sampling."""
    b, l = batch.shape

    # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
    u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
    indices = torch.arange(b, device=batch.device).float()
    t = (u0 + indices / b) % 1

    p_mask = (1 - sampling_eps) * t + sampling_eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    # Always keep BOS/EOS unmasked (official Dream eval behavior).
    mask_indices[:, 0] = False
    mask_indices[:, -1] = False

    noisy_batch = torch.where(mask_indices, mask_id, batch)

    return noisy_batch, p_mask


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 1.0:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    input_ids = batch
    if input_ids.device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids).logits
    else:
        logits = model(input_ids).logits
    # Since BOS is always unmasked, the first logits will not be used.
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    if cfg_scale > 1.0:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + cfg_scale * (logits - un_logits)
    return logits[:, :batch.shape[1]]


@torch.no_grad()
def get_nll_mc(
    model,
    prompt,
    answer,
    mc_num=8,
    cfg_scale=1.0,
    mask_id=151666,
    sampling_eps=1e-3,
    log_type="ftb",
):
    if prompt is None or len(prompt) == 0:
        seq = answer.unsqueeze(0)
        prompt_len = 0
    else:
        seq = torch.cat([prompt, answer], dim=0).unsqueeze(0)
        prompt_len = len(prompt)

    if log_type == "ftb":
        prompt_index = torch.arange(seq.shape[1], device=seq.device) < prompt_len
    elif log_type == "btf":
        prompt_index = torch.arange(seq.shape[1], device=seq.device) >= prompt_len
    else:
        prompt_index = torch.zeros(seq.shape[1], device=seq.device, dtype=torch.bool)

    losses = []
    for _ in range(mc_num):
        perturbed_seq = seq.clone()
        perturbed_seq_, p_mask = forward_process(seq, mask_id, sampling_eps)
        if log_type == "ftb":
            perturbed_seq[:, -len(answer):] = perturbed_seq_[:, -len(answer):]
        elif log_type == "btf":
            perturbed_seq[:, :prompt_len] = perturbed_seq_[:, :prompt_len]
        elif log_type == "union":
            perturbed_seq = perturbed_seq_
        else:
            raise ValueError(f"Unsupported log_type: {log_type}")
        mask_index = perturbed_seq == mask_id

        if not torch.any(mask_index):
            continue

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        if logits.shape[0] != seq.shape[0] or not torch.any(mask_index.squeeze()):
            continue

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
        loss = loss.sum()
        losses.append(loss.item())

    if not losses:
        return float('nan')

    return sum(losses) / len(losses)


def eval_ppl_dream(model, tokenizer, device, sampling_eps=1e-3):
    """Evaluate perplexity on wikitext-2 for Dream models."""
    # Get mask_id from config or tokenizer
    try:
        mask_id = model.config.mask_token_id
        if mask_id is None:
            mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise AttributeError
    except (AttributeError, KeyError):
        mask_id = 151666  # Dream default mask_token_id

    wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', streaming=True)
    wikitext_dataset = wikitext_dataset.take(128)

    text_list = [line for line in wikitext_dataset['text'] if line.strip()]
    text = "\n".join(text_list)

    input_ids = tokenizer.encode(text)
    print(f'token count: {len(input_ids)}')

    max_length = 1024
    chunk_size = 64
    stride = 64

    total_nll = 0.0
    total_tokens = 0
    num_samples = 0

    print(f"evaluating... (max_length={max_length}, chunk_size={chunk_size}, stride={stride})")

    for i in tqdm(range(chunk_size, len(input_ids) - chunk_size, stride)):
        prompt_start = max(0, i - (max_length - chunk_size))
        prompt_tokens = torch.tensor(input_ids[prompt_start:i]).to(device)

        answer_tokens = torch.tensor(input_ids[i : i + chunk_size]).to(device)

        if prompt_tokens.numel() == 0 or answer_tokens.numel() < chunk_size:
            continue

        nll = get_nll_mc(
            model,
            prompt_tokens,
            answer_tokens,
            mc_num=8,
            mask_id=mask_id,
            sampling_eps=sampling_eps,
            log_type="ftb",
        )

        if not np.isnan(nll):
            total_nll += nll
            total_tokens += answer_tokens.numel()
            num_samples += 1

    if total_tokens > 0:
        final_ppl = np.exp(total_nll / total_tokens)
        print(f"\nwikitext-2 PPL: {final_ppl:.4f} (Based on {num_samples} samples, {total_tokens} tokens)")
        return final_ppl
    else:
        print("No valid samples were found for PPL calculation.")
        return float('nan')

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = DreamModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    model.seqlen = 4096
    return model

def copy_dream_support_files(source_model, save_dir, cache_dir=None):
    if not source_model or not save_dir:
        return

    extra_files = (
        "generation_config.json",
        "configuration_dream.py",
        "modeling_dream.py",
        "generation_utils.py",
        "generation_utils_block.py",
        "tokenization_dream.py",
        "chat_template.jinja",
    )

    os.makedirs(save_dir, exist_ok=True)
    copied = []

    if os.path.isdir(source_model):
        for filename in extra_files:
            src_path = os.path.join(source_model, filename)
            dst_path = os.path.join(save_dir, filename)
            if os.path.isfile(src_path) and not os.path.isfile(dst_path):
                if os.path.abspath(src_path) == os.path.abspath(dst_path):
                    continue
                shutil.copy2(src_path, dst_path)
                copied.append(filename)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            print(f"Skipping extra file copy; huggingface_hub not available: {exc}")
            return

        for filename in extra_files:
            dst_path = os.path.join(save_dir, filename)
            if os.path.isfile(dst_path):
                continue
            try:
                src_path = hf_hub_download(
                    repo_id=source_model,
                    filename=filename,
                    cache_dir=cache_dir,
                )
            except Exception:
                continue
            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue
            shutil.copy2(src_path, dst_path)
            copied.append(filename)

    if copied:
        print("Copied extra Dream files to saved model: " + ", ".join(copied))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Dream model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "sink", "sink_sparsegpt",
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search", "dllm_unstruct", "dllm_struct"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--calib_dataset', type=str, default="wikitext2", help='Calibration dataset name.')
    parser.add_argument('--diffusion_steps', type=int, default=50, help='Total diffusion steps for calibration.')
    parser.add_argument('--sample_interval', type=int, default=5, help='Interval for sampling diffusion steps.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--noise_eps', type=float, default=1e-3)
    parser.add_argument('--sink_std', type=float, default=2.0)
    parser.add_argument('--sink_alpha', type=float, default=5.0)
    parser.add_argument('--lambda_mu', type=float, default=0.5)
    parser.add_argument('--lambda_sigma', type=float, default=0.2)

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading dllm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sink":
            prune_sink(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sink_sparsegpt":
            prune_sink_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    print("Evaluating perplexity on wikitext-2...")
    ppl = eval_ppl_dream(model, tokenizer, device, sampling_eps=args.noise_eps)
    print("-" * 50)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        copy_dream_support_files(args.model, args.save_model, cache_dir=args.cache_dir)

if __name__ == '__main__':
    main()
