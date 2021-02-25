import sys
import logging
import numpy as np
from libs.mongo_wrapper import MongoWrapper
from model.kogpt2 import get_tokenizer

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


config_path = 'config/db_config_filt_web.json'
mw = MongoWrapper(config_path)

tokenizer = get_tokenizer(vocab_file='./vocab/vocab_web/vocab_web-vocab.json',
                          merge_file='./vocab/vocab_web/vocab_web-merges.txt',
                          enable_postprocessiing=False,
                          enable_padding=False)

freq = []
nbatch = 10000
batch_text = []
for idx, item in enumerate(mw):
    if idx % nbatch == 0 and len(batch_text) != 0:
        out = tokenizer.encode_batch(batch_text)
        freq += [len(o.ids) for o in out]

        out = np.histogram(freq, bins=range(0, 130, 10))
        cnt_arr = out[0]
        bins = out[1]

        total = sum(cnt_arr)
        p_arr = [c / total for c in cnt_arr]
        logging.info("[Progress]: %d" % idx)
        logging.info("[Cnt]: %s" % cnt_arr)
        logging.info("[Prob]: %s" % p_arr)
        logging.info("[Bins]: %s" % bins)

        batch_text = []

    text = item[0]['data']['filt_text']
    batch_text.append(text)




