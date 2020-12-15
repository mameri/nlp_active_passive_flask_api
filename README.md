# A brief description of the project

In this project, we use the power of machine learning technique to classify the
sentences into active and passive.

## Machine learning

To perform the classification task, we use the Naive Bayes algorithm. 
The training data comes from the "2nd_Gore-Bush.txt" file. 
Since the data is not labeled, in the first stage, we use the parts of speech
tags. 
The sentence is tagged passive if a past participle verb succeeds by a be form.

Naive Bayes algorithm requires an input feature that specifies the required
information.
Since we only need grammatical representation for this task, for each sentence
parts of speech are extracted. 
If a verb is in the be form, that part is replaced with the "BE" tag.

The model consists of a count vectorizer followed by a Bayesian classifier.

 the experiments are conducted over five folds
of cross-validation To select the best hyperparameters.
Then the best model based on the f1 measure is stored to be used for the
prediction.

The model is then evaluated on the test data "3rd_Bush-Kerry.txt".

## Rest API 

Users can use the API endpoint ```/predict/sentence```
with the sentence to get the prediction result back.

# Installation

```git clone git@github.com:mameri/nlp_active_passive_flask_api.git

pip install -r requirements.txt
```

# Run Flask API

Run the flask app by: 

```
python app.py
```

Example usage Call API with 

```angular2html
curl -X POST -d  '{"sentence":"It is told to be true."}'  http://localhost:5000/predict/sentence
```
Its output:
```{"sentence":"It is told to be true.","type":"passive"}```

# Run Tests

```pytest```

# Run ML backend

Use this command to see the evaluation of the model on test data.
If the model does not exist, it creates the model.

```
python ml_backend.py data/2nd_Gore-Bush.txt data/3rd_Bush-Kerry.txt
```

Sample output where ```model.pkl``` exist.

```
precision:0.945054945054945, recall:0.86, accuracy:0.9910672308415609, f1:0.9005235602094239
```

Sample output when ```model.pkl``` does not exist.

```Fitting five folds for each of 315 candidates, totaling 1575 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done 1575 out of 1575 | elapsed:  2.2min finished
Best score: 0.862
Best parameters set:
	clf__alpha: 0.4
	vect__max_df: 0.15
	vect__ngram_range: (2, 2)
precision:0.945054945054945, recall:0.86, accuracy:0.9910672308415609, f1:0.9005235602094239
```


