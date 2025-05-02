from django import forms

class CSVUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file', help_text='Please upload a CSV file') 

class ModelTrainForm(forms.Form):
    MODEL_CHOICES = [
        ('logistic', 'Logistic Regression'),
        ('tree', 'Decision Tree'),
        ('svm', 'Support Vector Machine'),
    ]
    
    model_type = forms.ChoiceField(choices=MODEL_CHOICES, label="Model Type")
    test_size = forms.FloatField(min_value=0.1, max_value=0.9, initial=0.2, label="Test Size (0.1 to 0.9)")
