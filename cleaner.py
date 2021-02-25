import logging
import sys
import re
import json
import emoji
from libs.mongo_wrapper import MongoWrapper
from arguments import get_preprocessing_args

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)


def filter_text(sent):
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
    args = get_preprocessing_args()

    md = MongoWrapper(args.config_src)
    with open(args.config_src) as fp:
        config = json.load(fp)
    src_collection = config['COLLECTIONS'][0]
    logging.info("[Source]: %s" % src_collection)

    md_trgt = MongoWrapper(args.config_trgt)
    with open(args.config_trgt) as fp:
        config = json.load(fp)
    trgt_collection = config['COLLECTIONS'][0]
    logging.info("[Target]: %s" % trgt_collection)


    # TODO: 인덱싱 생성하기 idx
    docs = []
    cnt = 0
    for i in range(len(md)):
        # [{'data': dict, 'collection_name':str}]
        if i % 10000 == 0:
            logging.info("[Current]: %d " % i)
            if len(docs) != 0:
                md_trgt.insert_docs(docs, collection_name=trgt_collection)
                md_trgt.update_meta_info(collection_name=trgt_collection)
                docs = []

        item = md[i]
        doc = item[0]

        data = doc['data']
        filt_text = filter_text(data['form'])
        if len(filt_text) == 0 or filt_text == '.':
            continue

        out = {'filt_text': filt_text, 'idx': cnt}
        docs.append(out)
        cnt += 1




