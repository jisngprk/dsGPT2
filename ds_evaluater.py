import os, random, sys, argparse, time
import pathlib
import logging
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from config.gpt_config import kogpt2_config_345m
from model.kogpt2 import get_gpt2_model, get_tokenizer
from dataset.lm_dataset import MaskedLMDataset
from libs.mongo_wrapper import MongoWrapper
from arguments import get_ds_args
import deepspeed


logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


class DsModelLoader:
    def __init__(self, model, tokenizer, args):
        self.model = model

        self._init_distributed(args)
        self._init_model(args)
        self._set_random_seed(args.seed)

        model_engine, _, _ = self._setup_model_and_optimizer(args)
        self.model_engine = model_engine
        self.tokenizer = tokenizer

    def _init_distributed(self, args):
        """Initilaize distributed env"""
        logging.info("[INIT]: init distributed")

        device = None
        if args.local_rank is not None:
            device = args.local_rank

        torch.cuda.set_device(device)

    def _init_model(self, args):
        """Initilaize model gpu, fp16."""
        logging.info("[INIT]: allocate model to gpu")
        self.model.half()
        self.model.cuda(torch.cuda.current_device())

    def _set_random_seed(self, seed):
        """Set random seed for reproducability."""
        if seed is not None and seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _get_optimizer(self):

        # LayerNorm, Bias, Embedding
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return param_groups

    def _setup_model_and_optimizer(self, args):
        """Setup model and optimizer."""

        param_groups = self._get_optimizer()
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=param_groups,
            args=args
        )

        return model_engine, optimizer, lr_scheduler

    def load_from_checkpoint(self, args):
        logging.info("[Load checkpoint]: %s" % args.load_dir)
        logging.info("[Load checkpoint]: %s" % args.ckpt_id)
        self.model_engine.load_checkpoint(args.load_dir, args.ckpt_id)

    def _forward_step(self, record, args):
        token_ids, mask, label = record
        token_ids = torch.LongTensor(token_ids)
        mask = torch.LongTensor(mask)
        label = torch.LongTensor(label)

        token_ids = token_ids.to(args.local_rank) # (batch_size, max_len)
        mask = mask.to(args.local_rank)  # (batch_size, max_len)
        label = label.to(args.local_rank) # (batch_size, max_len)

        outputs = self.model_engine(token_ids) # (batch_size, max_len, embed_dims)
        logits = outputs.logits

        return logits

    def forward(self, tok_ids):
        input_data = tok_ids[:-1]
        label = tok_ids[1:]
        mask = np.where(input_data != self.tokenizer.token_to_id("<pad>"), 1, 0)

        input_data = np.array(input_data)
        mask = np.array(mask)
        label = np.array(label)
        record = (input_data, mask, label)
        logits = self._forward_step(record, args)

        print(logits.shape)
        logit = logits[-1, :]
        rand_indices = torch.randperm(10)
        top_logits, idxs = torch.topk(logit, k=10, dim=-1)
        last_tok_ids = idxs[rand_indices[0]]

        return last_tok_ids

    def generate_text(self, initial_sent):
        output = self.tokenizer.encode(initial_sent)

        tok_ids = output.ids
        tok_ids = [tokenizer.token_to_id("<s>")] + tok_ids
        print("-", tok_ids)
        tok_ids = np.array(tok_ids)

        cnt = 0
        while cnt < 50:
            last_tok_ids = self.forward(tok_ids)
            last_tok_ids = last_tok_ids.cpu().numpy()

            cnt += 1
            if last_tok_ids == tokenizer.token_to_id("</s>"):
                logging.info("fail")
                continue
            else:
                tok_ids = np.append(tok_ids, last_tok_ids)
                logging.info("success")


        return tok_ids


# https://towardsdatascience.com/text-generation-with-pretrained-gpt2-using-pytorch-563c7c90700
if __name__ == '__main__':
    args = get_ds_args()

    tokenizer = get_tokenizer(vocab_file='./vocab/vocab_web/vocab_web-vocab.json',
                              merge_file='./vocab/vocab_web/vocab_web-merges.txt',
                              enable_postprocessiing=False,
                              enable_padding=False)

    model = get_gpt2_model(config_dict=kogpt2_config_345m)

    # config_path = 'config/db_config_filt_web.json'
    # mw = MongoWrapper(config_path)
    # dataset = MaskedLMDataset(mw,
    #                           tokenizer)
    model_loader = DsModelLoader(model=model,
                                 tokenizer=tokenizer,
                                 args=args)
    model_loader.load_from_checkpoint(args)
    while True:
        text = input('input sent >')
        tok_ids = model_loader.generate_text(text)
        sent = model_loader.tokenizer.decode(tok_ids)
        print(sent)
