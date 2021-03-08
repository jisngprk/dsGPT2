import os
import json
from flask import Flask
from flask import make_response
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS, cross_origin
from arguments import get_ds_args
from model_loader import ModelLoader


class Generator(Resource):
    def __init__(self, env_args):
        self.env_args = env_args

        self.ml = ModelLoader(env_args)

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('sentence')
        args = parser.parse_args()
        print(args)
        sentence = args.get('sentence', "안녕")
        out_ids, out_sent, enc_sent, resp_sent = self.ml.generate(sentence)

        out_dict = {"input": sentence, "response": resp_sent}

        resp = make_response(json.dumps(out_dict), 200)
        resp.headers['Cache-Control'] = 'no-cache,no-store,must-revalidate'
        resp.headers.extend({
            "content-type": "application/json"
        })

        return resp


if __name__ == '__main__':
    args = get_ds_args()

    app = Flask(__name__)
    api = Api(app)
    CORS(app)

    api.add_resource(Generator, '/generate', resource_class_kwargs={'env_args': args})

    app.run(host='0.0.0.0', port=4000, threaded=False, debug=False)
