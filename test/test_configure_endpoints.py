from flask import Flask
import json
from test.confpytest import *

from configure_endpoints import configure_app

def test_root(model):
    app = Flask(__name__)
    configure_app(app, model)
    client = app.test_client()

    response = client.get('/')
    assert "up and running" == json.loads(response.get_data())
    assert response.status_code == 200

def test_predict_sentence(model):
    app = Flask(__name__)
    configure_app(app, model)
    client = app.test_client()

    sentence_data = {'sentence': 'People are stopped, and we have to do something about that.'}

    response = client.post('/predict/sentence',
                           data= json.dumps(sentence_data))

    assert 'passive' == json.loads(response.get_data())['type']
    assert sentence_data['sentence'] == json.loads(response.get_data())['sentence']
