import logging
import pathlib
import sys
from tokenizers import ByteLevelBPETokenizer

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


if __name__ == '__main__':
    tokenizer = ByteLevelBPETokenizer(
        "./vocab/vocab_web/vocab_web-vocab.json",
        "./vocab/vocab_web/vocab_web-merges.txt"
    )

    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"),
                             pad_token="<pad>",
                             length=128)
    input_str = "이날 행사는 방문객들의 편의를 위해 공식행사를 전면 폐지하고 해를 보며 한 해의 소원을 빌고 이를 축하하는 축포 발사와 BAT 코리아의 지원과 사천시 새마을회의 봉사로 떡국을 무료로 나눠주는 소망 떡국 나눠 먹기 행사로 진행했다."
    output = tokenizer.encode(input_str)
    ouput_str = tokenizer.decode(output.ids)
    print(input_str)
    print(len(output.tokens))
    print(ouput_str)

