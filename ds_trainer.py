import json
import os, random, sys, argparse, time
import pathlib
import logging
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from config.gpt_config import kogpt2_config_112m_half, kogpt2_config_112m, kogpt2_config_345m
from model.kogpt2 import get_gpt2_model, get_tokenizer
from dataset.lm_dataset import MaskedLMDataset
from libs.mongo_wrapper import MongoWrapper
from arguments import get_ds_args
import deepspeed


logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


class Trainer:
    def __init__(self, model, dataset, args):
        self.model = model

        self._init_distributed(args)
        self._init_model(args)
        self._load_ds_config_dict(args)
        self._set_random_seed(args.seed)

        model_engine, optimizer, lr_scheduler = self._setup_model_and_optimizer(args)
        self.model_engine = model_engine
        self.optimizer = optimizer

        tr_iter, val_iter,\
        tr_dataloader, val_dataloader, \
        tr_set, val_set,\
        tr_sampler, val_sampler = self._get_dataloader(dataset, args)

        self.tr_iter = tr_iter
        self.val_iter = val_iter
        self.tr_dataloader = tr_dataloader
        self.val_dataloader = val_dataloader
        self.tr_sampler = tr_sampler
        self.val_sampler = val_sampler

        self._get_loss(args)

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

    def _load_ds_config_dict(self, args):
        with open(args.deepspeed_config) as fp:
            ds_config = json.load(fp)

        self.ds_config = ds_config
        logging.info("[Load ds_config]: %s" % ds_config)

    def _set_random_seed(self, seed):
        """Set random seed for reproducability."""
        if seed is not None and seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _get_dataloader(self, dataset, args):
        def _collate_fn(batch):
            data = [item[0] for item in batch]
            mask = [item[1] for item in batch]
            label = [item[2] for item in batch]
            return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

        nsamples = len(dataset)
        num_train = int(nsamples * args.tr_ratio)
        num_val = nsamples - num_train
        tr_set, val_set = torch.utils.data.random_split(dataset,
                                                        [num_train, num_val])

        tr_sampler = torch.utils.data.distributed.DistributedSampler(tr_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

        # MongoDB is not fork-safe (num_workers=0)
        # Use batchsize decided by deepspeed engine (train_micro_batch_size_per_gpu)
        logging.info("[Train batch size per gpu]: %d" % self.model_engine._config.train_micro_batch_size_per_gpu)
        logging.info("[Gradient accumulation steps]: %d" % self.model_engine._config.gradient_accumulation_steps)
        logging.info("[Total batch size]: %d" % self.model_engine._config.train_batch_size)

        tr_dataloader = DataLoader(
            tr_set, batch_size=self.model_engine._config.train_micro_batch_size_per_gpu, num_workers=0,
            shuffle=False, collate_fn=_collate_fn, sampler=tr_sampler)

        val_dataloader = DataLoader(
            val_set, batch_size=args.eval_batch_size, num_workers=0,
            shuffle=False, collate_fn=_collate_fn, sampler=val_sampler)

        tr_iter = iter(tr_dataloader)
        val_iter = iter(tr_dataloader)

        logging.info("[num data]: train - %s, valid - %s" % (len(tr_set), len(val_set)))
        logging.info("[batch size]: train - %s, valid - %s" %
                     (len(tr_dataloader), len(val_dataloader)))

        return tr_iter, val_iter, \
               tr_dataloader, val_dataloader,\
               tr_set, val_set,\
               tr_sampler, val_sampler

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

    def _resume_data_loader(self, args):
        """Resume data loader iteration from specified step"""
        pass

    def _init_wandb(self, args):
        wandb.init(project=args.wandb_dir, reinit=True)
        wandb.config.update(args)
        wandb.config.update(self.ds_config)
        wandb.config.update(args.selected_config)
        wandb.watch(self.model_engine)

    def _get_loss(self, args):
        if args.loss_type == 'lm_loss':
            self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def _forward_step(self, record, args):
        token_ids, mask, label = record
        token_ids = token_ids.to(args.local_rank) # (batch_size, max_len)
        mask = mask.to(args.local_rank)  # (batch_size, max_len)
        label = label.to(args.local_rank) # (batch_size, max_len)

        outputs = self.model_engine(token_ids) # (batch_size, max_len, embed_dims)
        logits = outputs.logits
        loss = self.loss_function(logits.transpose(2, 1), label) # (batch_size, max_len)

        mask = mask.half()
        loss = loss.half()

        const_bool = torch.zeros(1, dtype=torch.bool).to(args.local_rank)
        max_args = torch.argmax(logits, dim=-1)
        correct = max_args == label
        masked_correct = torch.where(mask == 1, correct, const_bool)
        acc_avg = torch.true_divide(masked_correct.sum(), mask.sum())

        return logits, loss, mask, label, acc_avg

    def _backward_step(self, loss, mask, args):
        const = torch.zeros(1).to(args.local_rank).half()
        masked_loss = torch.where(mask == 1, loss, const)

        # Max integer of fp16 is 65536.0
        # The sum of loss should not be larger than the max integer
        # Thus, the loss is averaged for sequence length first and those values are averaged on batch size
        sub_loss = masked_loss.sum(dim=-1)
        sub_mask = mask.sum(dim=-1)
        sub_avg = sub_loss/sub_mask
        loss_avg = sub_avg.mean()
        self.model_engine.backward(loss_avg)

        return loss_avg

    def _train_step(self, record, args):
        self.model_engine.train()
        logit, loss, mask, label, acc_avg = self._forward_step(record, args)
        loss_avg = self._backward_step(loss, mask, args)
        self.model_engine.step()

        return loss, loss_avg, acc_avg

    def train(self, args):
        self._init_wandb(args)

        t_start = time.time()
        ntr_samples = len(self.tr_dataloader)
        wpath = pathlib.Path(args.ckpt_dir) / pathlib.Path(args.workspace)

        logging.info(args)

        if args.restart:
            _, client_state = self.model_engine.load_checkpoint(wpath, args.ckpt_id)
            epoch = client_state['epoch']
            step_cnt = client_state['step_cnt']
            loss_avg = client_state['loss_avg']
            logging.info("[Restart]")
            logging.info("[Restart]: epoch: %d" % epoch)
            logging.info("[Restart]: step_cnt: %d" % step_cnt)
            logging.info("[Restart]: loss_avg: %f" % loss_avg)
        else:
            epoch = 0
            step_cnt = 0
            logging.info("[Initial start]")

        self.tr_sampler.set_epoch(epoch)
        self.val_sampler.set_epoch(epoch)
        logging.info("[Total train iterations]: %d" % args.train_iters)
        while step_cnt < args.train_iters:
            try:
                record = next(self.tr_iter)
            except:
                epoch += 1
                self.tr_sampler.set_epoch(epoch)
                self.tr_iter = iter(self.tr_dataloader)
                record = next(self.tr_iter)
                logging.info("[Reload train data] %d" % step_cnt)

            loss, loss_avg, acc_avg = self._train_step(record, args)
            logging.info("[Loss] %f" % loss_avg)
            lr_state = [group['lr'] for group in self.optimizer.param_groups][0]
            wandb.log({
                "Epoch": epoch,
                "Step": step_cnt / ntr_samples,
                "Loss (train)": loss_avg,
                "Acc (train)": acc_avg,
                "lr": lr_state
            }, step=step_cnt)

            if step_cnt % 100 == 0:
                try:
                    val_record = next(self.val_iter)
                except Exception as e:
                    self.val_sampler.set_epoch(epoch)
                    self.val_iter = iter(self.val_dataloader)
                    val_record = next(self.val_iter)
                    logging.info("[Reload valid data] %d" % step_cnt)

                self.model_engine.eval()
                logit, loss, mask, label, acc_val = self._forward_step(val_record, args)
                loss_val = self._backward_step(loss, mask, args)
                wandb.log({
                    "Loss (eval)": loss_val,
                    "Acc (eval)": acc_val,
                    "Time (sec)": time.time() - t_start
                }, step=step_cnt)

            if step_cnt % 2000 == 0:
                fstring = 'epoch%d-step%d' % (epoch, step_cnt)
                self.model_engine.save_checkpoint(wpath, fstring, client_state={
                    'epoch': epoch,
                    'step': step_cnt,
                    'loss_avg': loss_avg
                })

                logging.info("[Rank - %d, MODEL SAVE]: %s" % (args.rank, os.path.join(wpath, fstring)))

            step_cnt += 1


if __name__ == '__main__':
    args = get_ds_args()

    selected_config = None
    if args.model_select == '112m':
        selected_config = kogpt2_config_112m
    elif args.model_select == '112m_half':
        selected_config = kogpt2_config_112m_half
    elif args.model_select == '345m':
        selected_config = kogpt2_config_345m

    if selected_config is None:
        logging.error("[Fail]: Select model type")
        raise NotImplementedError

    vocab_size = int(args.vocab_id_dir.split('_')[1])
    selected_config['vocab_size'] = vocab_size

    args.selected_config = selected_config
    logging.info("vocab size %s" % vocab_size)

    vocab_dir = pathlib.Path(args.vocab_load_dir) /pathlib.Path(args.vocab_id_dir)
    vocab_file = list(vocab_dir.glob("*-vocab.json"))[0]
    vocab_file = str(vocab_file)
    merge_file = list(vocab_dir.glob("*-merges.txt"))[0]
    merge_file = str(merge_file)

    tokenizer = get_tokenizer(vocab_file=vocab_file,
                              merge_file=merge_file,
                              enable_postprocessiing=True,
                              enable_padding=True,
                              max_len=selected_config['n_ctx'])

    model = get_gpt2_model(config_dict=selected_config)

    mw = MongoWrapper(args.config_train)
    dataset = MaskedLMDataset(mw,
                              tokenizer)
    trainer = Trainer(model, dataset, args)
    trainer.train(args)