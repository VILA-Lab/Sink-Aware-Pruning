import argparse
import os 
import sys
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from importlib.metadata import version
from tqdm import tqdm

from model import LLaDAModelLM

from lib.prune_llada import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, prune_sink, prune_sink_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot


def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape
    target_len = (l - prompt_index.sum()).item()
    
    if target_len <= 0:
        return batch, torch.zeros(b, l, device=batch.device)

    k = torch.randint(1, target_len + 1, (), device=batch.device)
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    input_ids = batch
    logits = model(input_ids).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@torch.no_grad()
def get_ppl(model, prompt, answer, mc_num=8, cfg_scale=0., mask_id=126336):
    seq = torch.cat([prompt, answer], dim=0).unsqueeze(0)
    prompt_index = torch.arange(seq.shape[1], device=seq.device) < len(prompt)

    losses = []
    for _ in range(mc_num):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id
        
        if not torch.any(mask_index):
            continue

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)
        
        if logits.shape[0] != seq.shape[0] or not torch.any(mask_index.squeeze()):
             continue

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='mean')
        losses.append(loss.item())

    if not losses:
        return float('nan')

    # current block PPL
    return np.exp(sum(losses) / len(losses))


def eval_ppl_llada(model, tokenizer, device):
    """Evaluate perplexity on wikitext-2 for LLaDA models."""
    try:
        mask_id = tokenizer.mask_token_id
        if mask_id is None: raise AttributeError
    except (AttributeError, KeyError):
        mask_id = 126336

    wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', streaming=True)
    wikitext_dataset = wikitext_dataset.take(128)

    text_list = [line for line in wikitext_dataset['text'] if line.strip()]
    text = "\n".join(text_list)
    
    input_ids = tokenizer.encode(text)
    print(f'token count: {len(input_ids)}')

    max_length = 1024 
    chunk_size = 64    
    stride = 64 

    perplexities = []
    
    print(f"evaluating... (max_length={max_length}, chunk_size={chunk_size}, stride={stride})")
    
    for i in tqdm(range(chunk_size, len(input_ids) - chunk_size, stride)):
        prompt_start = max(0, i - (max_length - chunk_size))
        prompt_tokens = torch.tensor(input_ids[prompt_start:i]).to(device)
        
        answer_tokens = torch.tensor(input_ids[i : i + chunk_size]).to(device)

        if prompt_tokens.numel() == 0 or answer_tokens.numel() < chunk_size:
            continue

        ppl = get_ppl(model, prompt_tokens, answer_tokens, mc_num=8, mask_id=mask_id)
        
        if not np.isnan(ppl):
            perplexities.append(ppl)

    num_samples = len(perplexities)
    if num_samples > 0:
        log_ppls = np.log(np.array(perplexities))
        final_ppl = np.exp(np.mean(log_ppls))
        print(f"\nwikitext-2 PPL: {final_ppl:.4f} (Based on {num_samples} samples)")
        return final_ppl
    else:
        print("No valid samples were found for PPL calculation.")
        return float('nan')

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = LLaDAModelLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    model.seqlen = model.config.max_sequence_length 
    return model

def copy_llada_support_files(source_model, save_dir, cache_dir=None):
    if not source_model or not save_dir:
        return

    os.makedirs(save_dir, exist_ok=True)
    copied = []

    local_model_dir = os.path.join(os.path.dirname(__file__), "model")
    if os.path.isdir(local_model_dir):
        dst_model_dir = os.path.join(save_dir, "model")
        os.makedirs(dst_model_dir, exist_ok=True)
        for filename in os.listdir(local_model_dir):
            src_path = os.path.join(local_model_dir, filename)
            if not os.path.isfile(src_path):
                continue
            if not filename.endswith((".py", ".jinja")):
                continue
            dst_path = os.path.join(dst_model_dir, filename)
            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue
            if os.path.isfile(dst_path):
                continue
            shutil.copy2(src_path, dst_path)
            copied.append(os.path.join("model", filename))

    extra_files = (
        "configuration_llada.py",
        "modeling_llada.py",
    )

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
            if copied:
                print("Copied extra LLaDA files to saved model: " + ", ".join(copied))
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
        print("Copied extra LLaDA files to saved model: " + ", ".join(copied))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaDA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "sink", "sink_sgpt",
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
        elif args.prune_method == "sink_sgpt":
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
    ppl = eval_ppl_llada(model, tokenizer, device)
    print("-" * 50)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        copy_llada_support_files(args.model, args.save_model, cache_dir=args.cache_dir)

if __name__ == '__main__':
    main()
