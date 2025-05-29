from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .forms import TextClassificationForm
import io
import pickle
import os
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'static', 'saved_pipeline.pkl')

def index(request):
    form = TextClassificationForm()
    test_accuracy = None
    best_params = None
    best_cv_score = None
    error_message = None
    

    if request.method == "POST":
        action = request.POST.get('action')
        form = TextClassificationForm(request.POST, request.FILES)

        if action == 'load':
            try:
                with open(MODEL_PATH, 'rb') as f:
                    loaded_pipeline = pickle.load(f)

                # Load default test data (for demo)
                df = pd.read_csv('static/IMDB Dataset.csv').sample(2000, random_state=42)
                df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
                X = df['review']
                y = df['label']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                y_pred = loaded_pipeline.predict(X_test)
                test_accuracy = f"{accuracy_score(y_test, y_pred):.2%}"
            except Exception as e:
                error_message = f"Failed to load pre-trained model: {str(e)}"

        elif form.is_valid() and action =='train':
            file = request.FILES['file']
            try: 
                df = pd.read_csv(file)
                df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

                X = df['review']
                y = df['label']

                vec_type = form.cleaned_data['vectorizer_choice']
                model_type = form.cleaned_data['model_choice']

                vectorizer = TfidfVectorizer(stop_words='english') if vec_type == 'tfidf' else CountVectorizer(stop_words='english')
        
                if model_type == 'logreg':
                    classifier = LogisticRegression(solver = 'liblinear')
                    param_grid = {
                        'vectorizer__max_features': [10000],
                        'vectorizer__ngram_range': [(1, 1), (1, 2)],
                        'clf__C': [0.01, 0.1, 1]
                    }
                elif model_type == 'svm':
                    classifier = SVC()
                    param_grid = {
                        'vectorizer__max_features': [10000],
                        'vectorizer__ngram_range': [(1, 1), (1, 2)],
                        'clf__C': [0.1, 1, 10],
                        'clf__kernel': ['linear', 'rbf']
                    }
                elif model_type == 'rf':
                    classifier = RandomForestClassifier()
                    param_grid = None
                else:
                    classifier = LogisticRegression()

                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('clf', classifier)
                ])
        
        
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True
                )

                if param_grid:
                    grid = GridSearchCV(
                        pipeline, 
                        param_grid = param_grid, 
                        cv = 2, 
                        scoring='accuracy', 
                        verbose = 1, 
                        n_jobs = 1 
                    )
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    y_pred = best_model.predict(X_test)
                    test_accuracy = f"{accuracy_score(y_test, y_pred):.2%}"
                    best_cv_score = f"{grid.best_score_:.2%}"
                    best_params = grid.best_params_
                    model_to_save = best_model
                else:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    test_accuracy = f"{accuracy_score(y_test, y_pred):.2%}"
                    model_to_save = pipeline
                
                with open(MODEL_PATH, 'wb') as f:
                        pickle.dump(model_to_save, f)

            except Exception as e:
                error_message = f"Error processing the given file: {str(e)}"
    
    return render(request, 'project2/train.html', {
        'form': form,
        'result': test_accuracy,
        'best_cv_score': best_cv_score,
        'best_params': best_params,
        'error': error_message,
    })