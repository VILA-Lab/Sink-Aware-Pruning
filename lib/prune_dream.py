import time 
import heapq 
import math
import torch 
import torch.nn as nn 
from tqdm import tqdm
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


class SinkCollector:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.mass = None

    def reset(self, batch_size: int, seq_len: int) -> None:
        self.mass = torch.zeros((batch_size, seq_len), device=self.device, dtype=torch.float32)

    def add(self, layer_mass: torch.Tensor) -> None:
        if self.mass is None:
            self.mass = layer_mass.detach().to(device=self.device, dtype=torch.float32)
        else:
            self.mass += layer_mass.detach().to(device=self.device, dtype=torch.float32)


class SinkActivationStats:
    def __init__(self, in_features: int, device: torch.device) -> None:
        self.accum = torch.zeros(in_features, device=device, dtype=torch.float32)

    def update(self, inp: torch.Tensor, sink_mask: torch.Tensor, weight: float) -> None:
        x = inp
        if x.dim() == 2 and sink_mask is not None:
            bsz, seq_len = sink_mask.shape
            if x.shape[0] == bsz * seq_len:
                x = x.view(bsz, seq_len, -1)

        if sink_mask is not None and x.dim() == 3:
            mask = (1.0 - sink_mask).unsqueeze(-1).to(dtype=x.dtype)
            x = x * mask

        flat = x.reshape(-1, x.shape[-1]).float()
        norm = torch.norm(flat, p=2, dim=0)
        self.accum += float(weight) * norm.to(self.accum.device)


class SinkActivationCollector:
    def __init__(self, layers: dict) -> None:
        self.stats = {
            name: SinkActivationStats(layer.weight.shape[1], layer.weight.device) for name, layer in layers.items()
        }
        self._sink_mask = None
        self._weight = 1.0
        self._mask_cache = {}

    def set_context(self, sink_mask: torch.Tensor, weight: float) -> None:
        self._sink_mask = sink_mask
        self._weight = float(weight)
        self._mask_cache = {}

    def clear_context(self) -> None:
        self._sink_mask = None
        self._mask_cache = {}

    def _get_mask(self, device: torch.device):
        if self._sink_mask is None:
            return None
        if device not in self._mask_cache:
            self._mask_cache[device] = self._sink_mask.to(device=device)
        return self._mask_cache[device]

    def hook(self, name: str):
        def fn(_, inp, __):
            if self._sink_mask is None:
                return
            x = inp[0]
            mask = self._get_mask(x.device)
            self.stats[name].update(x, mask, self._weight)

        return fn


def _batch_iter(samples, batch_size: int):
    buf = []
    for sample in samples:
        buf.append(sample[0])
        if len(buf) == batch_size:
            yield torch.cat(buf, dim=0)
            buf = []
    if buf:
        yield torch.cat(buf, dim=0)


def _forward_diffusion(input_ids: torch.Tensor, timestep: int, total_steps: int, mask_id: int, eps: float) -> torch.Tensor:
    """Dream-style forward process with uniform sampling and BOS/EOS preserved.

    Note: Dream's forward process samples t uniformly per example; timestep/total_steps
    are kept for API parity with LLaDA-style pruning and are not used to set t.
    """
    bsz, seq_len = input_ids.shape
    u0 = torch.rand(1, device=input_ids.device, dtype=torch.float32)
    indices = torch.arange(bsz, device=input_ids.device, dtype=torch.float32)
    t = (u0 + indices / bsz) % 1
    p_mask = (1.0 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, seq_len)
    mask = torch.rand_like(input_ids.float()) < p_mask
    if seq_len >= 2:
        mask[:, 0] = False
        mask[:, -1] = False
    return torch.where(mask, torch.tensor(mask_id, device=input_ids.device), input_ids)


def _compute_sink_mask(mass: torch.Tensor, std_factor: float, alpha: float) -> torch.Tensor:
    mean = mass.mean(dim=1, keepdim=True)
    std = mass.std(dim=1, keepdim=True, unbiased=False)
    threshold = mean + std_factor * std
    return torch.sigmoid(alpha * (mass - threshold))


def _get_input_device(model, fallback: torch.device) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for key in ("model.embed_tokens", "model.transformer.wte", "transformer.wte"):
            if key in model.hf_device_map:
                return torch.device(model.hf_device_map[key])
    return fallback


def _make_attention_mask(input_ids: torch.Tensor, pad_token_id: int):
    if pad_token_id is None or input_ids.dim() != 2:
        return None
    pad_mask = input_ids != pad_token_id
    if torch.all(pad_mask):
        return None
    # Additive mask: 0 for tokens, -inf for pads, shape (bsz, 1, 1, seq_len).
    neg_inf = torch.finfo(torch.float32).min
    mask = (1.0 - pad_mask.float()) * neg_inf
    return mask[:, None, None, :]


@torch.no_grad()
def prune_sink_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """Sink-aware pruning using SparseGPT's Hessian-based reconstruction."""
    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataset = getattr(args, "calib_dataset", "wikitext2")
    diffusion_steps = getattr(args, "diffusion_steps", 50)
    sample_interval = getattr(args, "sample_interval", 5)
    batch_size = getattr(args, "batch_size", 1)
    noise_eps = getattr(args, "noise_eps", 1e-3)
    sink_std = getattr(args, "sink_std", 2.0)
    sink_alpha = getattr(args, "sink_alpha", 5.0)

    if sample_interval <= 0 or diffusion_steps <= 0:
        raise ValueError("diffusion_steps and sample_interval must be positive.")

    sampled_steps = list(range(sample_interval, diffusion_steps + 1, sample_interval))

    try:
        mask_id = model.config.mask_token_id
        if mask_id is None:
            mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise AttributeError
    except (AttributeError, KeyError):
        mask_id = 151666

    print("loading calibration data")
    seqlen = getattr(model, "seqlen", getattr(model.config, "max_position_embeddings", None))
    if seqlen is None:
        raise ValueError("Model sequence length is not set.")
    dataloader, _ = get_loaders(
        dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    input_device = _get_input_device(model, device)
    sink_device = input_device
    if hasattr(model, "hf_device_map"):
        devices = {torch.device(dev) for dev in model.hf_device_map.values()}
        if len(devices) > 1:
            sink_device = torch.device("cpu")

    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

    blocks = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=input_device)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError

    blocks[0] = Catcher(blocks[0])
    for batch in dataloader:
        try:
            batch_ids = batch[0].to(input_device)
            attention_mask = _make_attention_mask(batch_ids, pad_token_id)
            model(batch_ids, attention_mask=attention_mask, use_cache=False)
        except ValueError:
            pass
    blocks[0] = blocks[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    sink_collector = SinkCollector(sink_device)

    print("Computing sink masks for calibration samples...")
    total_samples = len(dataloader) if hasattr(dataloader, "__len__") else args.nsamples
    total_batches = max(1, math.ceil(total_samples / batch_size))
    sink_masks = [None] * total_samples
    progress = tqdm(total=total_batches * len(sampled_steps), desc="Sink detection", unit="step")
    try:
        sample_idx = 0
        for batch in _batch_iter(dataloader, batch_size):
            batch = batch.to(input_device)
            batch_mask = _make_attention_mask(batch, pad_token_id)
            bsz = batch.shape[0]
            aggregated_mask = torch.zeros((bsz, seqlen), device=sink_device, dtype=torch.float32)
            total_weight = 0.0

            for step in sampled_steps:
                noisy = _forward_diffusion(batch, step, diffusion_steps, mask_id, noise_eps)
                sink_collector.reset(bsz, seqlen)
                _ = model(
                    input_ids=noisy,
                    attention_mask=batch_mask,
                    use_cache=False,
                    attn_collector=sink_collector.add,
                )
                sink_mask = _compute_sink_mask(sink_collector.mass, sink_std, sink_alpha)
                aggregated_mask += sink_mask
                total_weight += 1.0
                progress.update(1)

            aggregated_mask /= total_weight
            for i in range(bsz):
                if sample_idx >= total_samples:
                    break
                sink_masks[sample_idx] = aggregated_mask[i:i + 1]
                sample_idx += 1
    finally:
        progress.close()

    print("Pruning layers with SparseGPT + sink weighting...")
    for i in tqdm(range(len(blocks)), desc="Layer pruning"):
        layer = blocks[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        subset = find_layers(layer)
        gpts = {name: SparseGPT(subset[name]) for name in subset}

        def add_batch_sink(name, sink_weight):
            def tmp(_, inp, out):
                x = inp[0].data
                if len(x.shape) == 2:
                    x = x.unsqueeze(0)
                if sink_weight is not None:
                    w = sink_weight.unsqueeze(-1).to(x.device, x.dtype)
                    x = x * w
                gpt = gpts[name]
                if len(x.shape) == 3:
                    x = x.reshape((-1, x.shape[-1]))
                x = x.t()
                tmp_n = inp[0].shape[0] if len(inp[0].shape) == 3 else 1
                gpt.H *= gpt.nsamples / (gpt.nsamples + tmp_n)
                gpt.nsamples += tmp_n
                x = math.sqrt(2 / gpt.nsamples) * x.float()
                gpt.H += x.matmul(x.t())
            return tmp

        for j in range(args.nsamples):
            handles = []
            sink_mask = sink_masks[j].to(inps.device) if sink_masks[j] is not None else None
            sink_weight = (1.0 - sink_mask) if sink_mask is not None else None
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch_sink(name, sink_weight)))

            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

            for h in handles:
                h.remove()

        for name in gpts:
            print(f"Pruning layer {i} name {name}")
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        blocks[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sink(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataset = getattr(args, "calib_dataset", "wikitext2")
    diffusion_steps = getattr(args, "diffusion_steps", 50)
    sample_interval = getattr(args, "sample_interval", 5)
    batch_size = getattr(args, "batch_size", 1)
    noise_eps = getattr(args, "noise_eps", 1e-3)
    sink_std = getattr(args, "sink_std", 2.0)
    sink_alpha = getattr(args, "sink_alpha", 5.0)

    if sample_interval <= 0 or diffusion_steps <= 0:
        raise ValueError("diffusion_steps and sample_interval must be positive.")

    sampled_steps = list(range(sample_interval, diffusion_steps + 1, sample_interval))

    try:
        mask_id = model.config.mask_token_id
        if mask_id is None:
            mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise AttributeError
    except (AttributeError, KeyError):
        mask_id = 151666

    print("loading calibration data")
    seqlen = getattr(model, "seqlen", getattr(model.config, "max_position_embeddings", None))
    if seqlen is None:
        raise ValueError("Model sequence length is not set.")
    dataloader, _ = get_loaders(
        dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    blocks = model.model.layers
    layer_map = {}
    for i, block in enumerate(blocks):
        subset = find_layers(block)
        for name, layer in subset.items():
            layer_map[f"model.layers.{i}.{name}"] = layer

    act_collector = SinkActivationCollector(layer_map)
    hooks = [layer.register_forward_hook(act_collector.hook(name)) for name, layer in layer_map.items()]

    input_device = _get_input_device(model, device)
    sink_device = input_device
    if hasattr(model, "hf_device_map"):
        devices = {torch.device(dev) for dev in model.hf_device_map.values()}
        if len(devices) > 1:
            sink_device = torch.device("cpu")
    sink_collector = SinkCollector(sink_device)
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

    total_samples = len(dataloader) if hasattr(dataloader, "__len__") else args.nsamples
    total_batches = max(1, math.ceil(total_samples / batch_size))
    progress = tqdm(total=total_batches * len(sampled_steps), desc="Sink calibration", unit="step")
    try:
        for batch in _batch_iter(dataloader, batch_size):
            batch = batch.to(input_device)
            attention_mask = _make_attention_mask(batch, pad_token_id)
            for step in sampled_steps:
                noisy = _forward_diffusion(batch, step, diffusion_steps, mask_id, noise_eps)

                sink_collector.reset(noisy.shape[0], noisy.shape[1])
                _ = model(
                    input_ids=noisy,
                    attention_mask=attention_mask,
                    use_cache=False,
                    attn_collector=sink_collector.add,
                )
                sink_mask = _compute_sink_mask(sink_collector.mass, sink_std, sink_alpha)

                act_collector.set_context(sink_mask, 1.0)
                _ = model(input_ids=noisy, attention_mask=attention_mask, use_cache=False)
                act_collector.clear_context()
                progress.update(1)
    finally:
        progress.close()
        for h in hooks:
            h.remove()
        model.config.use_cache = use_cache

    for name, layer in layer_map.items():
        stats = act_collector.stats[name]
        W = layer.weight.data
        W_metric = torch.abs(W) * stats.accum.reshape(1, -1).to(W.device)
        W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
        prune_k = int(W_metric.shape[1] * args.sparsity_ratio)
        if prune_k <= 0:
            continue
        indices = torch.topk(W_metric, prune_k, dim=1, largest=False).indices
        W_mask.scatter_(1, indices, True)
        W[W_mask] = 0

    torch.cuda.empty_cache()
