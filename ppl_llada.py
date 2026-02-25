import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import os
import random
from tqdm import tqdm
import numpy as np

seed = 2025
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)

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


def main(path):
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(e)
        return

    try:
        mask_id = tokenizer.mask_token_id
        if mask_id is None: raise AttributeError
    except (AttributeError, KeyError):
        mask_id = 126336


    wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', streaming=True)
    # wikitext_dataset = load_dataset('Salesforce/wikitext', split='test', streaming=True)
    

    text_list = [line for line in wikitext_dataset['text'] if line.strip()]
    # text_list = text_list[:100]
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

        ppl = get_ppl(model, prompt_tokens, answer_tokens, mc_num=128, mask_id=mask_id)
        
        if not np.isnan(ppl):
            perplexities.append(ppl)

    num_samples = len(perplexities)
    if num_samples > 0:
        log_ppls = np.log(np.array(perplexities))
        final_ppl = np.exp(np.mean(log_ppls))
        print(f"\nwikitext-2 PPL: {final_ppl:.4f} (Based on {num_samples} samples)")
    else:
        print("No valid samples were found for PPL calculation.")
    
    print("-" * 50)


if __name__ == '__main__':
    models_to_evaluate = [
        'GSAI-ML/LLaDA-8B-Base',
        'GSAI-ML/LLaDA-8B-Instruct'
    ]
    
    for model_path in models_to_evaluate:
        main(model_path)