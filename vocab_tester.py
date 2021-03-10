import logging
import pathlib
import sys

from arguments import get_ds_args
from model.kogpt2 import get_tokenizer

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


if __name__ == '__main__':
    args = get_ds_args()

    vocab_dir = pathlib.Path(args.vocab_load_dir) /pathlib.Path(args.vocab_id_dir)
    vocab_file = list(vocab_dir.glob("*-vocab.json"))[0]
    vocab_file = str(vocab_file)
    merge_file = list(vocab_dir.glob("*-merges.txt"))[0]
    merge_file = str(merge_file)

    tokenizer = get_tokenizer(vocab_file=vocab_file, merge_file=merge_file, enable_padding=True, max_len=128)

    input_str = "이날은 날씨가 좋다"
    output = tokenizer.encode(input_str)
    output_str = tokenizer.decode(output.ids)
    print(output_str)


