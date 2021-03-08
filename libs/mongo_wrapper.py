from torch.utils.data import Dataset
import pymongo
import json
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class MongoWrapper:
    """
    Load single turn Q,A data

    """
    def __init__(self, config_path, filter_func=None):
        """
        1. MongoDB collection들을 통합된 인덱스로 접근할 수 있음
        2. 개별 collection의 idx는 개수, 순서, 유니크를 보장해야함

        :param config_path: db config 경로
        """

        with open(config_path) as fp:
            db_config = json.load(fp)
        self.db_config = db_config
        self.filter_func = filter_func

        conn_str = db_config['MONGO_CONNECTION_STRING']
        con_db = db_config['MONGO_CONNECTION_DB']
        collection_list = db_config['COLLECTIONS']
        self.connection = pymongo.MongoClient(conn_str)
        self.db = self.connection.get_database(con_db)
        self.collections = self._load_collections(collection_list)
        self.meta_info = self._load_metainfo(collection_list)
        self.ndoc = None
        logging.info("[Mongo]: Loaded %s" % self.meta_info)

    def __len__(self):
        if not self.ndoc:
            ndoc = 0
            for value in self.meta_info.values():
                ndoc += value['num_docs']

            self.ndoc = ndoc

        return self.ndoc

    def __getitem__(self, idx):
        docs = []
        if isinstance(idx, slice):
            for nidx in range(idx.start, idx.stop):
                collection_name, idx = self._convert_idx(nidx)
                data = self.collections[collection_name].find({'idx': idx})[0]

                if self.filter_func:
                    data = self.filter_func(data)

                doc = {'data': data, 'collection_name': collection_name}
                docs.append(doc)
            return docs
        else:
            collection_name, idx = self._convert_idx(idx)
            data = self.collections[collection_name].find({'idx': idx})[0]

            if self.filter_func:
                data = self.filter_func(data)

            doc = {'data': data, 'collection_name': collection_name}
            docs.append(doc)
            return docs

    def _load_collections(self, collection_list):
        if not isinstance(collection_list, list):
            collection_list = [collection_list]

        collections = dict()
        for col in collection_list:
            collections[col] = self.db[col]
            logger.info("[Mongo]: %s is loaded" % col)
        return collections

    def _load_metainfo(self, collection_list):
        meta_info_conn = self.db['meta_info']
        meta_info = OrderedDict()
        for item in list(meta_info_conn.find({})):
            if item['collection_name'] not in collection_list:
                continue

            collection_name = item['collection_name']
            sub_dict = {'num_docs': item['num_docs']}
            meta_info.update({collection_name: sub_dict})

        prev = 0
        for name, info in meta_info.items():
            sub_info = {'sidx': prev, 'eidx': prev + info['num_docs']}
            prev = prev + info['num_docs']
            info.update(sub_info)

        return meta_info

    def _convert_idx(self, idx):
        """
        collection 따라서 idx 를 변환하기
        :param idx:
        :return:
        """
        collection_name = None
        for name, info in self.meta_info.items():
            if idx >= info['sidx'] and idx < info['eidx']:
                idx = idx - info['sidx']
                collection_name = name
                break

        return collection_name, idx

    def _get_update_op(self, doc, fields):
        if not isinstance(fields, list):
            fields = [fields]

        set_dict = dict()
        for f in fields:
            set_dict[f] = doc[f]

        return pymongo.UpdateOne({'_id': doc['_id']}, {"$set": set_dict}, upsert=True)

    def _get_insert_op(self, doc):
        return pymongo.InsertOne(doc)

    def update_docs(self, docs, fields):
        if not isinstance(docs, list):
            docs = [docs]

        ops = []
        for doc in docs:
            op = self._get_update_op(doc, fields)
            ops.append(op)

        return ops

    def insert_docs(self, docs, collection_name):
        if collection_name not in self.collections:
            raise KeyError

        if not isinstance(docs, list):
            docs = [docs]

        ops = []
        for doc in docs:
            op = self._get_insert_op(doc)
            ops.append(op)

        # logging.info(ops[:10])
        self.collections[collection_name].bulk_write(ops, ordered=False)

    def update_meta_info(self, collection_name):
        is_update = False
        if collection_name in self.meta_info:
            is_update = True

        total_docs = self.collections[collection_name].count_documents({})
        logging.info("[Update]: collection - %s " % collection_name)
        logging.info("[Update]: total docs - %s " % total_docs)
        logging.info("[Update]: meta info - %s " % is_update)

        if is_update:
            self.db['meta_info'].update_one({'collection_name': collection_name},
                                            {'$set':{'num_docs': total_docs}})
        else:
            self.db['meta_info'].insert_one({'collection_name': collection_name,
                                             'num_docs': total_docs})
            collection_list = self.db_config['COLLECTIONS']
            self.meta_info = self._load_metainfo(collection_list)

    def export_to_file(self, fpath, collection_name):
        logging.info("[Export]: %s" % fpath)

        info = self.meta_info[collection_name]
        info = dict(info)
        num_docs = int(info['num_docs'])
        with open(fpath, 'w') as fp:
            text_lines = []
            for idx in range(num_docs):
                doc = self.__getitem__(idx)[0]
                text = doc['data']['filt_text']
                text += '\n'
                text_lines.append(text)
                if idx % 10000 == 0:
                    fp.writelines(text_lines)
                    text_lines = []
                    logging.info("[Write]: %d" % idx)

    def create_single_index(self, collection_name, index_name, order=1):
        self.collections[collection_name].create_index([(index_name, order)])