import sys
import logging
import re
import torch
from model.kogpt2 import get_gpt2_model, get_tokenizer, extract_vocab_path, load_gpt2_config, extract_ckpt_path
from arguments import get_ds_args

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


class ModelLoader:
    def __init__(self, args):
        self.args = args

        vocab_size, vocab_file, merge_file = extract_vocab_path(args)
        selected_config = load_gpt2_config(args)
        self._set_device(args)
        self._load_tokenizer(vocab_file, merge_file, args)
        self._load_model(selected_config, args)

        self.user_info = dict()

    def _set_device(self, args):
        # TODO: TEST GPU load
        if args.use_cpu:
            logging.info("[Device]: CPU %d is selected" % args.gpu_id)
            device = torch.device("cpu")
        else:
            logging.info("[Device]: GPU %d is selected" % args.gpu_id)
            # torch.cuda.set_device(args.gpu_id)
            device = torch.device("cuda:%d" % args.gpu_id)

        self.device = device

    def _load_tokenizer(self, vocab_file, merge_file, args):
        self.tokenizer = get_tokenizer(vocab_file=vocab_file,
                                       merge_file=merge_file,
                                       enable_padding=args.enable_padding,
                                       enable_bos=args.enable_bos,
                                       enable_eos=args.enable_eos,
                                       max_len=args.truncated_len)

    def _load_model(self, selected_config, args):
        args.selected_config = selected_config
        self.model = get_gpt2_model(config_dict=selected_config)

        ckpt_path = extract_ckpt_path(args)
        if args.use_cpu:
            out = torch.load(ckpt_path, map_location=self.device)
            state_dict = out['module']
            self.model.load_state_dict(state_dict)
        else:
            out = torch.load(ckpt_path)
            state_dict = out['module']
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

        self.model.eval()
        logging.info("[Load]: load model, ckpt %s" % ckpt_path)

    def _enroll_user(self, session_key, user_name):
        self.user_info[session_key] = user_name

    def _handle_replace_str(self, sentence):
        sentence = re.sub("name[0-9]+", "", sentence)
        return sentence

    def generate(self, sentence):
        logging.info("[Run]: generate sentence")
        logging.info("[Input]: %s" % sentence)

        sentence = sentence.strip()
        enc = self.tokenizer.encode(sentence)

        if self.args.train_mode == 'finetune':
            enc_ids = [self.tokenizer.token_to_id('<usr>')] + enc.ids + [self.tokenizer.token_to_id('<sys>')] \
                    + [self.tokenizer.token_to_id('<s>')]
        else:
            enc_ids = enc.ids

        logging.info(enc_ids)
        enc_tensor = torch.LongTensor([enc_ids])

        if self.args.train_mode == 'pretrain':
            logging.info("[Run]: pretrained model")
            out = self.model.generate(enc_tensor,
                                      pad_tokien_id=self.tokenizer.token_to_id('<pad>'),
                                      bos_token_id=self.tokenizer.token_to_id('<s>'),
                                      eos_token_id=self.tokenizer.token_to_id('</s>'),
                                      min_length=self.args.min_length,
                                      max_length=self.args.max_length,
                                      do_sample=self.args.do_sample,
                                      top_k=self.args.top_k,
                                      temperature=self.args.temperature,
                                      repetition_penalty=self.args.repetition_penalty)
        elif self.args.train_mode == 'finetune':
            logging.info("[Run]: finetuned model")
            out = self.model.generate(enc_tensor,
                                      pad_tokien_id=self.tokenizer.token_to_id('<pad>'),
                                      bos_token_id=self.tokenizer.token_to_id('<s>'),
                                      eos_token_id=self.tokenizer.token_to_id('</s>'),
                                      min_length=self.args.min_length,
                                      max_length=self.args.max_length,
                                      do_sample=self.args.do_sample,
                                      top_p=self.args.top_p,
                                      temperature=self.args.temperature,
                                      repetition_penalty=self.args.repetition_penalty)
        else:
            raise NotImplementedError

        print(self.args.min_length)
        print(self.args.max_length)
        out_ids = out[0].tolist()
        out_sent = self.tokenizer.decode(out_ids)
        out_sent = self._handle_replace_str(out_sent)
        enc_sent = self.tokenizer.decode(enc_ids)

        if self.args.train_mode == 'pretrain':
            resp_sent = out_sent
        elif self.args.train_mode == 'finetune':
            resp_sent = out_sent.replace(enc_sent, "")
        else:
            raise NotImplementedError

        resp_sent = resp_sent.replace("<s>", "")
        resp_sent = resp_sent.replace("</s>", "")
        resp_sent = resp_sent.replace("<sys>", "")
        resp_sent = resp_sent.replace("<usr>", "")
        resp_sent = resp_sent.replace("<pad>", "")

        return out_ids, out_sent, enc_sent, resp_sent


# https://towardsdatascience.com/text-generation-with-pretrained-gpt2-using-pytorch-563c7c90700
if __name__ == '__main__':
    args = get_ds_args()

    ml = ModelLoader(args)
    while True:
        input_sent = input("input > ")
        out_ids, out_sent, enc_sent, resp_sent = ml.generate(input_sent)

        print("-")
        print(out_ids)
        print(out_sent)
        print(enc_sent)
        print(resp_sent)
