from django import forms


class CSVUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file',
                           help_text='Please upload a CSV file')


class ModelTrainForm(forms.Form):
    MODEL_CHOICES = (
        ('rf',  'Random Forest'),
        ('svm', 'Support Vector Machine'),
        ('xgb', 'Extreme Gradient Boost'),
    )

    # Always required
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        required=True,
        initial='rf',
        label="Model Type",
    )

    # REQUIRED with strict range + HTML attributes so browsers enforce it too
    test_size = forms.FloatField(
        min_value=0.1,
        max_value=0.9,
        initial=0.2,
        label="Test Size (0.1 to 0.9)",
        widget=forms.NumberInput(attrs={
            "step": "0.01",          # <-- key: allow decimals cleanly
            "min": "0.1",
            "max": "0.9",
            "inputmode": "decimal",
            "placeholder": "0.20",   # optional: avoids early :invalid styling in some UAs
            "required": True
        }),
        error_messages={
            "required": "Please enter a test size.",
            "min_value": "Test size must be at least 0.10.",
            "max_value": "Test size must be at most 0.90.",
            "invalid": "Enter a valid number between 0.10 and 0.90."
        }
    )

    # ----- Random Forest (all OPTIONAL) -----
    rf_n_estimators = forms.IntegerField(
        min_value=10, max_value=500, initial=100,
        label="Number of Estimators", required=False,
        help_text="Number of trees in the forest"
    )
    rf_max_depth = forms.IntegerField(
        min_value=1, max_value=50, initial=5,
        label="Max Depth", required=False,
        help_text="Maximum depth of the trees"
    )
    RF_MAX_FEATURES_CHOICES = (
        ('sqrt', 'Square Root'),
        ('log2', 'Log2'),
        # NOTE: string "None" (convert to None in the view)
        ('None', 'All Features'),
    )
    rf_max_features = forms.ChoiceField(
        choices=RF_MAX_FEATURES_CHOICES, initial='sqrt',
        label="Max Features", required=False,
        help_text="Number of features to consider for best split"
    )

    # ----- SVM (all OPTIONAL) -----
    SVM_KERNEL_CHOICES = (
        ('rbf', 'RBF'),
        ('linear', 'Linear'),
        ('poly', 'Polynomial'),
        ('sigmoid', 'Sigmoid'),
    )
    svm_kernel = forms.ChoiceField(
        choices=SVM_KERNEL_CHOICES, initial='rbf',
        label="Kernel", required=False,
        help_text="Kernel type to be used in the algorithm"
    )
    svm_C = forms.FloatField(
        min_value=0.1, max_value=10.0, initial=1.0,
        label="C (Regularization)", required=False,
        help_text="Regularization parameter"
    )
    svm_gamma = forms.FloatField(
        min_value=0.001, max_value=1.0, initial=0.1,
        label="Gamma", required=False,
        help_text="Kernel coefficient"
    )
    # If you ever want 'scale'/'auto', switch this to CharField and handle in the view.

    # ----- XGBoost (all OPTIONAL) -----
    xgb_n_estimators = forms.IntegerField(
        min_value=10, max_value=500, initial=100,
        label="Number of Estimators", required=False,
        help_text="Number of boosting rounds"
    )
    xgb_learning_rate = forms.FloatField(
        min_value=0.001, max_value=1.0, initial=0.01,
        label="Learning Rate", required=False,
        help_text="Step size shrinkage used to prevent overfitting"
    )
    xgb_max_depth = forms.IntegerField(
        min_value=1, max_value=15, initial=3,
        label="Max Depth", required=False,
        help_text="Maximum depth of a tree"
    )
