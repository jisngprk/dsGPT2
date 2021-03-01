"""
Download mongoDB collection to text files and train tokenizer
"""
import os
import logging
import sys
import multiprocessing
import random

from libs.mongo_wrapper import MongoWrapper
from arguments import get_preprocessing_args

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)

# NUM_PROCESS = 33
# DIR_PATH = './data_files/vocab_web'
# TRAIN_FILE_NAME = 'vocab_train'


class Downloader(multiprocessing.Process):
    def __init__(self, _id, idx_list, fpath, config_path):
        super().__init__()
        self._id = str(_id)
        self._idx_list = idx_list
        self.fpath = fpath
        self.config_path = config_path
        self.dataset = None

    def run(self):
        self.dataset = MongoWrapper(self.config_path)

        with open(self.fpath, 'w') as fp:
            text_lines = []
            for count, idx in enumerate(self._idx_list):
                doc = self.dataset[idx][0]
                text = doc['data']['filt_text']
                text += '\n'
                text_lines.append(text)
                if count % 1000 == 0:
                    fp.writelines(text_lines)
                    text_lines = []
                    logging.info("[Write (pid: %s)]: %d" % (self._id, count))

            if text_lines:
                fp.writelines(text_lines)


if __name__ == '__main__':
    args = get_preprocessing_args()
    logging.info(args)

    if not os.path.exists(args.vocab_train_dir):
        logging.info("[Make dir]: %s" % args.vocab_train_dir)
        os.makedirs(args.vocab_train_dir)

    md = MongoWrapper(args.config_src)

    ndata = len(md)
    idx_list = list(range(ndata))
    idx_list = random.sample(idx_list, args.nsample)
    nstep = args.nsample//args.num_process

    logging.info("[Download]: %d samples" % args.nsample)
    fname_list = []
    mplist = []
    for i in range(args.num_process+1):
        file_length = len(idx_list[nstep*i:nstep*(i+1)])
        logging.info("[File length]: %d" % file_length)
        if file_length == 0:
            continue

        fpath = os.path.join(args.vocab_train_dir, args.vocab_train_fname + str(i) + '.txt')
        cleaner = Downloader(_id=i,
                             idx_list=idx_list[nstep*i:nstep*(i+1)],
                             fpath=fpath,
                             config_path=args.config_src)
        cleaner.daemon = True
        mplist.append(cleaner)
        fname_list.append(fpath)

    for mp in mplist:
        mp.start()

    for mp in mplist:
        mp.join()