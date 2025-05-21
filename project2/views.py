from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def index(request):
    test_accuracy = None
    best_params = None
    best_cv_score = None

    if request.method == "POST":
        df = pd.read_csv('static/IMDB Dataset.csv')
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        X = df['review']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(solver='liblinear', max_iter=1000)),
        ])

        """param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [10000, 20000],
            'clf__C': [0.01, 0.1, 1, 10],
        }"""

        param_grid = {
            'tfidf__ngram_range': [(1, 1)],          # Just unigrams for speed | [(1, 1), (1, 2)]
            'tfidf__max_features': [10000],          # Smaller vocab size | [10000, 20000]
            'clf__C': [1],                           # One regularization value | [0.01, 0.1, 1, 10]
        }

        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv = 2, #make it back to 3 later
            scoring='accuracy', 
            verbose = 1, 
            n_jobs = 1 #make it back to -1 later
        )
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        test_accuracy = f"{accuracy:.2%}"
        best_cv_score = f"{grid.best_score_:.2%}"
        best_params = grid.best_params_

    return render(request, 'project2/train.html', {
        'result': test_accuracy,
        'best_cv_score': best_cv_score,
        'best_params': best_params
    })