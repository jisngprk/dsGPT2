import logging
import pathlib
import sys
from tokenizers import ByteLevelBPETokenizer
from arguments import get_preprocessing_args

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)

# VOCAB_DIR = './vocab/vocab_web'
# TRAIN_DIR = './data_files/vocab_web'


def train(paths, save_dir, prefix, nspecial, vocab_size=52000, min_frequency=2):
    unused_tokens = []
    for i in range(nspecial):
        unused_tokens.append("<unused%d>" % i)

    special_tokens = []
    special_tokens.append("<s>")
    special_tokens.append("<pad>")
    special_tokens.append("</s>")
    special_tokens.append("<unk>")
    special_tokens.append("<mask>")
    special_tokens.append("<sys>")
    special_tokens.append("<usr>")
    special_tokens.extend(unused_tokens)

    tokenizer = ByteLevelBPETokenizer()

    logging.info("[Train vocab]: %s/%s" % (save_dir, prefix))
    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    logging.info("[Save vocab]: %s/%s" % (save_dir, prefix))
    tokenizer.save_model(save_dir, prefix)


if __name__ == '__main__':
    args = get_preprocessing_args()
    logging.info(args)

    pathlib.Path(args.vocab_dir).mkdir(parents=True, exist_ok=True)
    logging.info("[Directory]: %s" % args.vocab_dir)

    paths = [str(x) for x in pathlib.Path(args.vocab_train_dir).glob("*.txt")]
    train(paths=paths,
          save_dir=args.vocab_dir,
          prefix=args.vocab_prefix,
          nspecial=100,
          vocab_size=32000)





