from django import forms

class TrainingForm(forms.Form):
    vectorizer_choice = forms.ChoiceField(choices=[
        ('tfidf', 'TF-IDF'),
        ('bow', 'Bag-of-Words'),
    ])
    classifier_choice = forms.ChoiceField(choices=[
        ('logreg', 'Logistic Regression'),
        ('svm', 'SVM')
    ])