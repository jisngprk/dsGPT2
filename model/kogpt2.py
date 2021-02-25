import sys
import logging
import json
from transformers import GPT2LMHeadModel, GPT2Config
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from config.gpt_config import kogpt2_config_345m

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


def get_gpt2_model(config_dict):
    logging.info("[Load GPT2]: %s" % config_dict)
    config = GPT2Config.from_dict(config_dict)
    model = GPT2LMHeadModel(config)

    return model


def get_tokenizer(vocab_file,
                  merge_file,
                  enable_postprocessiing=True,
                  enable_padding=True,
                  max_len=128):
    logging.info("[Load Vocab]: %s" % vocab_file)
    logging.info("[Load Vocab]: %s" % merge_file)
    tokenizer = ByteLevelBPETokenizer(
        vocab_file,
        merge_file
    )

    logging.info("[Add special token]: <s> - %d" % tokenizer.token_to_id("<s>"))
    logging.info("[Add special token]: </s> - %d" % tokenizer.token_to_id("</s>"))
    if enable_postprocessiing:
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )

    if enable_padding:
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"),
                                 pad_token="<pad>",
                                 length=max_len + 1)
        tokenizer.enable_truncation(max_length=max_len + 1)

    return tokenizer


if __name__ == '__main__':
    tokenizer = get_tokenizer(vocab_file='./vocab/vocab_web/vocab_web-vocab.json',
                              merge_file='./vocab/vocab_web/vocab_web-merges.txt', enable_padding=True, max_len=128)

    model = get_gpt2_model(config_dict=kogpt2_config_345m)
    print(model)
