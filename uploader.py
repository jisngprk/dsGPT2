import json, sys, logging, re
import pathlib
from libs.mongo_wrapper import MongoWrapper
from arguments import get_preprocessing_args
import emoji

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
    args = get_preprocessing_args()

    md = MongoWrapper(args.config_src)
    with open(args.config_src) as fp:
        config = json.load(fp)
    src_collection = config['COLLECTIONS'][0]
    logging.info("[Source]: %s" % src_collection)

    flist = pathlib.Path('./data_files/nikl_web').glob('*.json')

    pat = re.compile("ES")
    idx = 0
    for fpath in flist:
        if len(pat.findall(str(fpath))) != 0:
            continue
        data = json.load(fpath.open())
        docs = data['document']
        text_set = []
        for doc in docs:
            paragraph = doc['paragraph']
            for p in paragraph:
                text = p['form']
                filt_text = filter_text(text)
                if len(filt_text) < 5 or filt_text == '.':
                    continue

                data = {'filt_text': filt_text, 'idx': idx}
                text_set.append(data)
                idx += 1

        md.insert_docs(text_set, collection_name=src_collection)
        md.update_meta_info(collection_name=src_collection)


        # docs = data['document']
        # text_set = []
        # for doc in docs:
        #     ut_list = doc['utterance']
        #     for u in ut_list:
        #         text = u['form']
        #         data = {'form': text, 'idx': idx}
        #         text_set.append(data)
        #         idx += 1
        #
        # md.insert_docs(text_set, collection_name=src_collection)
        # md.update_meta_info(collection_name=src_collection)
