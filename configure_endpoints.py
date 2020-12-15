from flask import request, jsonify
import json
from ml_backend import *

def configure_app(app, model):

    @app.route('/')
    def root():
        return jsonify('up and running')

    @app.route('/predict/sentence', methods=['POST'])
    def predict_sentence():
        headers = request.headers

        if request.method == 'POST':
            data = json.loads(request.get_data())
            data['type'] = predict(model, data['sentence'])
            return jsonify(data)

        return b'Not found', 404

