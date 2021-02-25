import json, sys, logging
import pathlib
from libs.mongo_wrapper import MongoWrapper
from arguments import get_preprocessing_args

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)

if __name__ == '__main__':
    args = get_preprocessing_args()

    md = MongoWrapper(args.config_src)
    with open(args.config_src) as fp:
        config = json.load(fp)
    src_collection = config['COLLECTIONS'][0]
    logging.info("[Source]: %s" % src_collection)

    flist = pathlib.Path('./data_files/nikl_spoken').glob('*.json')

    idx = 0
    for fpath in flist:
        data = json.load(fpath.open())
        docs = data['document']
        text_set = []
        for doc in docs:
            ut_list = doc['utterance']
            for u in ut_list:
                text = u['form']
                data = {'form': text, 'idx': idx}
                text_set.append(data)
                idx += 1

        md.insert_docs(text_set, collection_name=src_collection)
        md.update_meta_info(collection_name=src_collection)
