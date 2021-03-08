from torch.utils.data import Dataset
import sys
import logging
import numpy as np

from arguments import get_ds_args
from libs.mongo_wrapper import MongoWrapper
from model.kogpt2 import get_tokenizer, extract_vocab_path
from dataset.filter_funcs import conv_filter


logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


class ConvDataset(Dataset):
    def __init__(self, mw, tokenizer, max_len):
        self._data = mw

        self.tokenizer = tokenizer
        self.usr_token_id = tokenizer.token_to_id('<usr>')
        self.sys_token_id = tokenizer.token_to_id('<sys>')
        self.pad_token_id = tokenizer.token_to_id('<pad>')
        self.bos_id = tokenizer.token_to_id('<s>')
        self.eos_id = tokenizer.token_to_id('</s>')
        self.maskt_id = tokenizer.token_to_id('<mask>')

        self.usr_token = '<usr>'
        self.sys_token = '<sys>'
        self.bos = '<s>'
        self.eos = '</s>'
        self.maskt = '<mask>'

        self.max_len = max_len

        self.initial_log = True
        logging.info("[Success]: Load dataset")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        doc = self._data[idx]
        doc = doc[0]
        turn = doc['data']
        q = turn['q']
        a = turn['a']

        # <usr> <s> .. </s>
        usr_ids = [
            self.usr_token_id,
        ] + self.tokenizer.encode(q).ids
        usr_len = len(usr_ids)

        # <sys> <s> .. </s>
        sys_ids = [
            self.sys_token_id,
        ] + self.tokenizer.encode(a).ids
        sys_len = len(sys_ids)

        prev_usr_len = usr_len
        prev_sys_len = sys_len

        if usr_len + sys_len > self.max_len:
            sys_len = self.max_len - usr_len
            if sys_len <= 0:
                usr_ids = usr_ids[-(int(self.max_len/2)):]
                usr_len = len(usr_ids)
                sys_len = self.max_len - usr_len
                assert sys_len > 0
            sys_ids = sys_ids[:sys_len]
            sys_len = len(sys_ids)
            assert sys_len == len(sys_ids), f'{sys_len} ==? {len(sys_ids)}'
            # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]

        # <usr> <s> .. </s> <sys> <s> .. </s> <pad> ...
        # training from <s> input token
        input_data = usr_ids + sys_ids + [self.pad_token_id] * (self.max_len - usr_len - sys_len)
        mask = [0] * (usr_len + 1) + [1] * (sys_len - 1) + [0] * (self.max_len - usr_len - sys_len)
        label = [self.maskt_id] * (usr_len + 1) + \
            sys_ids[2:] + \
            [self.pad_token_id] * (self.max_len - usr_len - sys_len + 1)

        if len(input_data) > 128 or len(mask) > 128 or len(label) > 128:
            print(len(input_data), len(mask), len(label))
            print(len(usr_ids), len(sys_ids), self.max_len)
            print(prev_usr_len, prev_sys_len)
            print('--')

        input_data = np.array(input_data)
        mask = np.array(mask)
        label = np.array(label)
        if self.initial_log:
            logging.info("[Input]: %s" % input_data)
            logging.info("[Mask]: %s" % mask)
            logging.info("[Label]: %s" % label)

            self.initial_log = False

        return input_data, mask, label


if __name__ == '__main__':
    args = get_ds_args()
    vocab_size, vocab_file, merge_file = extract_vocab_path(args)

    tokenizer = get_tokenizer(vocab_file=vocab_file,
                              merge_file=merge_file,
                              enable_padding=args.enable_padding,
                              enable_bos=args.enable_bos,
                              enable_eos=args.enable_eos,
                              max_len=args.truncated_len)

    config_path = './config/db_config_finetune.json'
    mw = MongoWrapper(config_path,
                      filter_func=conv_filter)
    dataset = ConvDataset(mw=mw,
                          tokenizer=tokenizer,
                          max_len=128)
    data = dataset[100]
    print(data)