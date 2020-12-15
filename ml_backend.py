import sys, os, re
import pickle
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def tokenize(training_file_name):
    with open(training_file_name, 'r') as f_h:
        data = f_h.read()
    relines = re.split('[.?,;-]{1}', data)
    return [line.strip().lower() for line in relines if len(line) > 0]

def preprocess_sentence(sentence):

    be_form = {'am', 'is', 'are', 'be', 'been', 'being', 'was', 'were', "'m", "'s", "'m"}
    blob = TextBlob(sentence)
    is_passive = 0
    previous_verb = ''
    words = []
    for i, tag in enumerate(blob.tags):
        if previous_verb in be_form and tag[1] == 'VBN':
            is_passive = 1
        previous_verb = tag[0] if tag[1].startswith('VB') else previous_verb
        if tag[0] in be_form:
            words.append('BE')
        else:
            words.append(tag[1] )
    return " ".join(words), is_passive

def extract_tags(sentences):

    labels = []
    data = []
    for sentence in sentences:
        words, is_passive = preprocess_sentence(sentence)
        data.append(words)
        labels.append(is_passive)
    return data, labels

def pre_process_data(file_name):
    sentences = tokenize(file_name)
    data, labels = extract_tags(sentences)
    return data, labels


def evaluate(model, sentences, labels):
    predicted_labels = model.predict(sentences)

    mat = confusion_matrix(labels, predicted_labels)
    precision  = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    accuracy = accuracy_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)

    print(
        'precision:{}, recall:{}, accuracy:{}, f1:{} '.format(precision, recall,
                                                              accuracy, f1))

    return precision, recall, accuracy, f1


    # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
    #             xticklabels=set(labels), yticklabels=set(predicted_labels))
    # plt.xlabel('true label')
    # plt.ylabel('predicted label')
    # plt.savefig('confusion.png')

def fine_tune_nb_model(sentences, labels):

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])
    parameters = {
        'vect__max_df': (0.15, 0.25, 0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2),(1, 3),(2,2),(2, 3), (3,3), (1,4)),  # unigrams or bigrams
        'clf__alpha': (1.0,0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    }

    grid_search = GridSearchCV(pipeline,  parameters, cv=5, verbose=1,
                               return_train_score=True, scoring='f1')

    grid_search.fit(sentences, labels)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return grid_search.best_estimator_



def create_model(training_file = 'data/2nd_Gore-Bush.txt'):

    if not os.path.exists('model.pkl') :
        train_data, train_labels = pre_process_data(training_file)
        model = fine_tune_nb_model(train_data, train_labels)
        pickle.dump(model, open("model.pkl", 'wb'))

    model = pickle.load(open("model.pkl", 'rb'))
    return model

def evaluate_mode_on_file(model, test_file_name):
    test_data, test_labels = pre_process_data(test_file_name)
    return evaluate(model, test_data, test_labels)

def predict(model, sentence):
    return 'active' if model.predict([preprocess_sentence(sentence)[0]]) == 0 else 'passive'

if __name__ == "__main__":

    if len(sys.argv) > 1:
        training_file_name = sys.argv[1]
        test_file_name = sys.argv[2]
        model = create_model(training_file_name)

        evaluate_mode_on_file(model, test_file_name)
        evaluate_mode_on_file(model, training_file_name)



