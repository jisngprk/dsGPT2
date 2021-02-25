from torch.utils.data import Dataset
import sys
import logging

from libs.mongo_wrapper import MongoWrapper
from model.kogpt2 import get_tokenizer
import numpy as np

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


class MaskedLMDataset(Dataset):
    def __init__(self, mw, tokenizer):
        self._data = mw
        self.tokenizer = tokenizer

        self.initial_log = True
        logging.info("[Success]: Load dataset")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        doc = self._data[idx]
        doc = doc[0]
        filt_text = doc['data']['filt_text']
        output = self.tokenizer.encode(filt_text)
        tok_ids = output.ids
        tok_ids = np.array(tok_ids)

        input_data = tok_ids[:-1]
        label = tok_ids[1:]
        mask = np.where(input_data != self.tokenizer.token_to_id("<pad>"), 1, 0)

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
    tokenizer = get_tokenizer(vocab_file='./vocab/vocab_web/vocab_web-vocab.json',
                              merge_file='./vocab/vocab_web/vocab_web-merges.txt', enable_padding=True, max_len=128)

    config_path = './config/db_config_filt_web.json'
    mw = MongoWrapper(config_path)
    dataset = MaskedLMDataset(mw=mw,
                              tokenizer=tokenizer)
    data = dataset[100000]
    print(data)