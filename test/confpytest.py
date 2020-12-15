import pytest
import pickle
from flask import Flask
from app import *


@pytest.fixture
def model():
    model = pickle.load(open("model.pkl", 'rb'))
    return model

@pytest.fixture
def get_temp_dir(tmpdir_factory):
    directory = tmpdir_factory.mktemp('text_files')
    return directory

@pytest.fixture
def get_token_file(get_temp_dir):
    token_file = get_temp_dir + '/test_tokenize.txt'
    with open(token_file, 'w') as f:
        f.write('This is a sample sentence. It is attached on the wall.')
    return token_file

@pytest.fixture
def get_token_text():
    return ['this is a sample sentence', 'it is attached on the wall']


@pytest.fixture
def app(model):
    app = Flask(__name__)
    client = app.test_client()
    return client

