from ml_backend import *
from test.confpytest import *

def test_tokenize(get_token_file, get_token_text):
    tokens = tokenize(get_token_file)
    assert tokens == get_token_text


def test_preprocess_sentence(get_token_text):

    assert (preprocess_sentence(get_token_text[0])) == ('DT BE DT JJ NN', 0)
    assert (preprocess_sentence(get_token_text[1])) == ('PRP BE VBN IN DT NN', 1)

def test_extract_tags(get_token_text):
    data , labels = extract_tags(get_token_text)

    assert len(data) == len(get_token_text)
    assert len(labels) == len(get_token_text)

def test_process_data(get_token_file, get_token_text):
    data, labels = pre_process_data(get_token_file)
    assert len(data) == len(get_token_text)
    assert len(labels) == len(get_token_text)

def test_evaluate(model, get_token_file):
    data, labels = pre_process_data(get_token_file)
    eval_result = evaluate(model, data, labels)
    assert len(eval_result) == 4
    assert eval_result[3] > 0.5

def test_create_model():
    model = create_model('data/2nd_Gore-Bush.txt')
    result = (model.predict([(preprocess_sentence(
        'People are stopped, and we have to do something about that.')[
        0])]))
    assert 1 == result

def test_evaluate_model_on_file(model):
    eval_result = evaluate_mode_on_file(model, 'data/3rd_Bush-Kerry.txt')
    assert len(eval_result) == 4
    assert eval_result[3] > 0.5

def test_predict(model):
    result = predict(model, 'People are stopped, and we have to do something about that.')
    assert 'passive' == result
