import logging
import sys
import re
import json
import emoji
from libs.mongo_wrapper import MongoWrapper
from dataset.filter_funcs import conv_filter
from arguments import get_clean_args
from model.kogpt2 import get_tokenizer, extract_vocab_path

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


def filter_text(sent):
    # replace chinese character
    # replace '[ㅏ - ㅣ]'
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅎ가-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    rm_pattern = re.compile(r'#+')

    sent = pattern.sub('', sent)
    sent = url_pattern.sub('', sent)
    sent = rm_pattern.sub('', sent)
    sent = sent.strip()

    return sent


if __name__ == '__main__':
    args = get_clean_args()

    vocab_size, vocab_file, merge_file = extract_vocab_path(args)
    tokenizer = get_tokenizer(vocab_file=vocab_file,
                              merge_file=merge_file,
                              enable_padding=args.enable_padding,
                              enable_bos=args.enable_bos,
                              enable_eos=args.enable_eos,
                              max_len=args.truncated_len)

    md = MongoWrapper(args.config_src, filter_func=conv_filter)
    with open(args.config_src) as fp:
        config = json.load(fp)
    src_collection = config['COLLECTIONS'][0]
    logging.info("[Source]: %s" % src_collection)

    md_trgt = MongoWrapper(args.config_trgt, filter_func=conv_filter)
    with open(args.config_trgt) as fp:
        config = json.load(fp)
    trgt_collection = config['COLLECTIONS'][0]
    logging.info("[Target]: %s" % trgt_collection)


    # TODO: 인덱싱 생성하기 idx
    docs = []
    cnt = 0
    for i in range(len(md)):
        # [{'data': dict, 'collection_name':str}]
        if i % 10000 == 0 and i != 0:
            logging.info("[Current]: %d " % i)
            if len(docs) != 0:
                md_trgt.insert_docs(docs, collection_name=trgt_collection)
                md_trgt.update_meta_info(collection_name=trgt_collection)
                docs = []

        item = md[i]
        doc = item[0]

        data = doc['data']
        q = data['q']
        a = data['a']
        q_filt = filter_text(q)
        a_filt = filter_text(a)

        q_enc = tokenizer.encode(q)
        a_enc = tokenizer.encode(a)

        if len(q_filt) < 5 or q_filt == '.':
            continue
        if len(a_filt) < 5 or len(a_enc.ids) < 2 or a_filt == '.':
            # if len(a_enc.ids) < 2:
            #     print(q_filt, ':', a_filt, ':', a_enc.ids, ':', a_enc.tokens, ':', tokenizer.decode(a_enc.ids))
            continue

        out = {'q': q_filt, 'a': a_filt, 'idx': cnt}

        # Filter for Language Model data
        # filt_text = filter_text(data['form'])
        # if len(filt_text) < 5 or filt_text == '.':
        #     continue
        # out = {'filt_text': filt_text, 'idx': cnt}

        docs.append(out)
        cnt += 1




