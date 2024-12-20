import time

import torch
import torch.nn as nn
import tqdm
from gptq import *
from modelutils import *
from quant import *
from datasets import load_dataset

def find_layers_mamba(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class WikitextEvaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))
    
class LambadaEvaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
    
class TextGenerator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def generate_text(self, model, input_text, max_length=50):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        output = model.generate(inputs=input_ids,
                                max_length=max_length,
                                do_sample=True,
                                top_k=30,
                                pad_token_id=self.tokenizer.eos_token_id,
                                attention_mask=input_ids.new_ones(input_ids.shape))
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def calculate_perplexity(self, model, text):
        encodings = self.tokenizer(text, return_tensors='pt').to(self.device)
        input_ids = encodings.input_ids
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

        neg_log_likelihood = outputs.loss
        ppl = torch.exp(neg_log_likelihood)
        return ppl

def get_mamba(model_id):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.seqlen = 1024
    return model, tokenizer

@torch.no_grad()
def mamba_sequential(model, dataloader, dev, args):
    print('Starting GPTQ quantization for Mamba model...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.backbone.layers

    model.backbone.embeddings = model.backbone.embeddings.to(dev)
    model.backbone.norm_f = model.backbone.norm_f.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError

    print(layers[0])
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass 

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.backbone.embeddings = model.backbone.embeddings.cpu()
    model.backbone.norm_f = model.backbone.norm_f.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    print('Ready to quantize layers.')

    quantizers = {}
    for i, layer in enumerate(layers):
        print(f'Quantizing layer {i}...')
        layer = layer.to(dev)

        # find mamba layers
        subset = find_layers_mamba(layer)

        gptq = {}
        for name, sublayer in subset.items():
            gptq[name] = GPTQ(sublayer)
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        for h in handles:
            h.remove()

        for name, gptq_obj in gptq.items():
            print(f'Layer {i}, Sub-layer {name}: Quantizing...')
            gptq_obj.fasterquant(
                percdamp=args.percdamp,
                groupsize=args.groupsize,
                actorder=args.act_order,
                static_groups=args.static_groups
            )
            quantizer_name = f'model.backbone.layers.{i}.{name}'
            quantizers[quantizer_name] = gptq_obj.quantizer
            gptq_obj.free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layer = layer.cpu()
        layers[i] = layer
        del layer
        del gptq
        torch.cuda.empty_cache()


        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()

    # model, tokenizer = get_opt(args.model)
    model, tokenizer = get_mamba(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = mamba_sequential(model, dataloader, DEV, args)
        # quantizers = opt_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    print(model)
    dataset = load_dataset("lambada", split="validation[:100]")
    lambada_evaluator = LambadaEvaluator(dataset, tokenizer, "cuda")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikitext_evaluator = WikitextEvaluator(dataset, tokenizer, "cuda")
    text_generator = TextGenerator(tokenizer, 'cuda')

    original_text = text_generator.generate_text(model.to('cuda'), "I have a dream")
    print(original_text)

    print("\nLAMBADA Accuracy Evaluation")
    acc_original = lambada_evaluator.evaluate(model)
    print(f"accuracy on LAMBADA: {acc_original}")

    print("\nPerplexity Evaluation on WikiText")
    pp_wikitext = wikitext_evaluator.evaluate(model)
    print(f'perplexity on wikitext: {pp_wikitext}')

    print("\nPerplexity Evaluation on custom text")
    original_text = text_generator.generate_text(model, "I have a dream")
    print(f"output text:\n{original_text}")
    ppl = text_generator.calculate_perplexity(model, original_text)
    print(f"quantized model perplexity: {ppl.item():.2f}")

    # state-spaces/mamba-790m-hf
    # state-spaces/mamba-1.4b-hf
    # state-spaces/mamba-2.8b-hf

    total_bytes = 0

    for name, param in model.named_parameters():
        param_size = param.numel()  # Total number of elements in the parameter tensor
        param_bytes = 0

        # Check if the parameter belongs to a linear layer
        if "proj" in name:
            total_bytes += param_size * 3/8  # 1 byte for 8-bit quantized
        else:
            total_bytes += param_size * 4  # 4 bytes for 32-bit floating point

    print(f'Total model footprint in bytes: {total_bytes}')

    # Example usage: CUDA_VISIBLE_DEVICES=0 python mamba.py state-spaces/mamba-1.4b-hf c4 --wbits 8

