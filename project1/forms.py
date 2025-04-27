from django import forms

class CSVUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file', help_text='Please upload a CSV file') 