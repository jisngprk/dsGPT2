import logging
import pathlib
import sys
from tokenizers import ByteLevelBPETokenizer

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

    tokenizer = get_tokenizer(vocab_file=vocab_file,
                              merge_file=merge_file,
                              enable_postprocessiing=True,
                              enable_padding=True,
                              max_len=128)

    input_str = "이날 행사는 방문객들의 편의를 위해 공식행사를 전면 폐지하고 해를 보며 한 해의 소원을 빌고 이를 축하하는 축포 발사와 BAT 코리아의 지원과 사천시 새마을회의 봉사로 떡국을 무료로 나눠주는 소망 떡국 나눠 먹기 행사로 진행했다."
    output = tokenizer.encode(input_str)
    ouput_str = tokenizer.decode(output.ids)
    print(input_str)
    print(len(output.tokens))
    print(ouput_str)

