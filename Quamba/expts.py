import json
from utils import (
    build_mamba_and_tokenizer, 
    set_deterministic, 
    parse_options
)
import logging
import sys
import os
from quamba.eval_utils import eval_mamba_zero_shot, evaluate_ppl
import torch
from time import perf_counter

import torch.nn as nn
import tqdm
from datasets import load_dataset

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
        # Encode the text
        encodings = self.tokenizer(text, return_tensors='pt').to(self.device)

        # Define input_ids and target_ids
        input_ids = encodings.input_ids
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

        # Loss calculation
        neg_log_likelihood = outputs.loss

        # Perplexity calculation
        ppl = torch.exp(neg_log_likelihood)

        return ppl

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

import torch

class HellaSwagEvaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Tokenize the dataset
        def preprocess_function(examples):
            # Combine the context and endings
            first_sentences = [[context] * 4 for context in examples["ctx_a"]]
            question_headers = examples["ctx_b"]
            second_sentences = [
                [question_header + " " + ending for ending in endings]
                for question_header, endings in zip(question_headers, examples["endings"])
            ]

            # Flatten the lists for tokenization
            first_sentences = sum(first_sentences, [])
            second_sentences = sum(second_sentences, [])

            # Tokenize
            tokenized = self.tokenizer(
                first_sentences,
                second_sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Reshape to match the format for multiple-choice models
            return {
                "input_ids": tokenized["input_ids"].view(-1, 4, tokenized["input_ids"].shape[-1]),
                "attention_mask": tokenized["attention_mask"].view(-1, 4, tokenized["attention_mask"].shape[-1]),
                "labels": examples["label"]
            }

        self.dataset = self.dataset.map(preprocess_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, correct = 0, 0

        for batch in self.dataset:
            # Move input tensors to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: [batch_size, num_choices]
            predictions = logits.argmax(dim=-1)  # Get the index of the highest logit

            # Calculate accuracy
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        accuracy = correct / total
        return accuracy


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
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(self.device)
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
    
def main(args):    
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0] # Assume that the models name is like "model_type-<model_size, model version>"
    assert model_name != None, "Please check the model path."
    logging.info(f"Creating Model:{model_name}")
    model, tokenizer = build_mamba_and_tokenizer(args, model_type)
    model.config.use_cache = False
    dataset = load_dataset("lambada", split="validation[:100]")
    lambada_evaluator = LambadaEvaluator(dataset, tokenizer, "cuda")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikitext_evaluator = WikitextEvaluator(dataset, tokenizer, "cuda")
    text_generator = TextGenerator(tokenizer, 'cuda')
    logs = {}
    if args.eval_fp16:
        if args.eval_ppl:
            logging.info(f"Evaluating ppl result (fp16), dataset: {args.ppl_dataset}")
            ppl_results = evaluate_ppl(model, tokenizer, model_name, batch_size=args.batch_size, device="cuda", dataset=args.ppl_dataset)
            logs['ppl'] = ppl_results
        if args.eval_zero_shot:
            logging.info(f"Evaluating Result using lm_eval (fp16), task(s): {args.task_list}")
            lm_eval_results = eval_mamba_zero_shot(
                model, tokenizer, 
                model_type=model_type,
                task_list=args.task_list,
                batch_size=args.batch_size,
                fewshot=args.fewshot,
                limit=100 if args.testing else None
            )
            logs['lm_eval'] = lm_eval_results['results']
        if not args.eval_ppl and not args.eval_zero_shot:
            logging.warn("No task to run with, try `--eval_ppl`, `--eval_zero_shot`?")
        if args.log_dir:
            logs['args'] = vars(args)
            logs['quantization_config'] = 'fp16'
            os.makedirs(args.log_dir, exist_ok=True)
            log_paths = os.path.join(args.log_dir, f"{model_name}_fp16.json")
            with open(log_paths, 'a') as fp:
                logging.info(f"Saving result to {log_paths}")
                json.dump(logs, fp, indent=4)
        return

    """
    Start evaluating Quantized Models from here
    """
    logging.info(f"evaluating quantization type: {args.quant_type}")
    if args.quant_type.lower() == "real":
        from quamba.real_quant.modelutils_mamba import quantize_model_mamba as rlq_mamba
        model_quamba = rlq_mamba(model, model_type, tokenizer, "cuda", args)
    elif args.quant_type.lower() == "fake":
        from quamba.fake_quant.modelutils_mamba import quantize_model_mamba as faq_mamba
        model_quamba = faq_mamba(model, model_type, tokenizer, "cuda", args)
    else:
        logging.error(f"Unrecognized quantization type: {args.quant_type}")
    model_quamba.eval()

    print("\n\n LAMBADA EVAL")
    print("\nLAMBADA Accuracy Evaluation")
    acc_original = lambada_evaluator.evaluate(model_quamba)
    print(f"accuracy on LAMBADA: {acc_original}")


    print("\n\n WIKITEXT EVAL")
    print("\nPerplexity Evaluation on WikiText")
    t1_start = perf_counter() 
    pp_wikitext = wikitext_evaluator.evaluate(model_quamba)
    t1_end = perf_counter() 
    print(f'perplexity on wikitext: {pp_wikitext}')
    print("Elapsed time:", t1_end - t1_start) 


if __name__ =='__main__':    
    set_deterministic(1234)
    args = parse_options()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    main(args)

