# Installation

```git clone git@github.com:mameri/nlp_active_passive_flask_api.git

pip install -r requirements.txt
```

# Run Flask api

Run the flask app by: 

```
python app.py
```

Example usage Call api with 

```angular2html
curl -X POST -d  '{"sentence":"It is told to be true."}'  http://localhost:5000/predict/sentence
```
It output:
```{"sentence":"It is told to be true.","type":"passive"}```

# Run Tests

```pytest```

# Run ML backend

Use this command to see the evaluation of model on test data.
If model does not exist, it creates the model.

```
python ml_backend.py data/2nd_Gore-Bush.txt data/3rd_Bush-Kerry.txt
```

Sample output where ```model.pkl``` exist.

```
precision:0.945054945054945, recall:0.86, accuracy:0.9910672308415609, f1:0.9005235602094239
```

Sample output when ```model.pkl``` does not exist.

```Fitting 5 folds for each of 315 candidates, totalling 1575 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done 1575 out of 1575 | elapsed:  2.2min finished
Best score: 0.862
Best parameters set:
	clf__alpha: 0.4
	vect__max_df: 0.15
	vect__ngram_range: (2, 2)
precision:0.945054945054945, recall:0.86, accuracy:0.9910672308415609, f1:0.9005235602094239
```


