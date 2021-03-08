import sys
import logging
import pathlib
from transformers import GPT2LMHeadModel, GPT2Config
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from config.gpt_config import kogpt2_config_112m_half, kogpt2_config_112m, kogpt2_config_345m

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


def load_gpt2_config(args):
    selected_config = None
    if args.model_select == '112m':
        selected_config = kogpt2_config_112m
    elif args.model_select == '112m_half':
        selected_config = kogpt2_config_112m_half
    elif args.model_select == '345m':
        selected_config = kogpt2_config_345m

    if selected_config is None:
        logging.error("[Fail]: Select model type")
        raise NotImplementedError

    return selected_config


def get_gpt2_model(config_dict):
    logging.info("[Load GPT2]: %s" % config_dict)
    config = GPT2Config.from_dict(config_dict)
    model = GPT2LMHeadModel(config)

    return model


def extract_ckpt_path(args):
    wpath = pathlib.Path(args.ckpt_dir) / pathlib.Path(args.workspace)
    ckpt_path = wpath / pathlib.Path(args.ckpt_id)
    print(list(wpath.glob("*")))
    print(list(ckpt_path.glob("*")))
    ckpt_fpath = list(ckpt_path.glob("*.pt"))[0]
    if not ckpt_fpath:
        logging.error("[Fail]: Load ckpt")
    else:
        logging.info("[Success]: Load %s" % ckpt_fpath)

    return ckpt_fpath


def extract_vocab_path(args):
    vocab_dir = pathlib.Path(args.vocab_load_dir) / pathlib.Path(args.vocab_id_dir)
    vocab_file = list(vocab_dir.glob("*-vocab.json"))[0]
    vocab_file = str(vocab_file)
    merge_file = list(vocab_dir.glob("*-merges.txt"))[0]
    merge_file = str(merge_file)

    vocab_size = int(args.vocab_id_dir.split('_')[1])

    logging.info("[Vocab info]: %s" % vocab_size)
    logging.info("[Vocab info]: %s" % vocab_file)
    logging.info("[Vocab info]: %s" % merge_file)
    return vocab_size, vocab_file, merge_file


def get_tokenizer(vocab_file, merge_file, enable_padding=True, enable_bos=True, enable_eos=True, max_len=128):
    logging.info("[Load Vocab]: %s" % vocab_file)
    logging.info("[Load Vocab]: %s" % merge_file)
    tokenizer = ByteLevelBPETokenizer(
        vocab_file,
        merge_file
    )


    if enable_bos and enable_eos:
        logging.info("[Load Vocab]: Enable BOS, EOS")
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )
    elif enable_bos and not enable_eos:
        logging.info("[Load Vocab]: Enable BOS only")
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A",
            pair="<s> $A $B",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
            ],
        )
    elif not enable_bos and not enable_eos:
        logging.info("[Load Vocab]: Disable BOS, EOS")
    else:
        logging.debug("Fail condition for bos, eos")
        raise NotImplementedError

    logging.info("[Load Vocab]: Enable truncation")
    tokenizer.enable_truncation(max_length=max_len)


    if enable_padding:
        logging.info("[Load Vocab]: Enable padding")
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"),
                                 pad_token="<pad>",
                                 length=max_len)
    else:
        logging.info("[Load Vocab]: Disable padding")

    return tokenizer


if __name__ == '__main__':
    tokenizer = get_tokenizer(vocab_file='./vocab/vocab_web/vocab_web-vocab.json',
                              merge_file='./vocab/vocab_web/vocab_web-merges.txt', enable_padding=True, enable_bos=True,
                              enable_eos=True, max_len=128)

    model = get_gpt2_model(config_dict=kogpt2_config_345m)
    print(model)
