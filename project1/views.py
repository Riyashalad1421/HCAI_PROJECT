from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import json
from .forms import CSVUploadForm


#Very basic view as an example
def index(request):
    return render(request, 'project1/index.html')


def upload_csv(request):
    error = None
    csv_data = None
    headers = None
    data = None
    plots = []
    row_count = 0
    filename = None
    features_json = None
    data_json = None
    target_classes_json = None
    unique_classes_json = None
    features = None
    
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Read the CSV file
                csv_file = request.FILES['file']
                
                # Get the filename without extension
                filename = csv_file.name
                if filename.endswith('.csv'):
                    filename = filename[:-4]  # Remove .csv extension
                
                # Check if it's a CSV file
                if not csv_file.name.endswith('.csv'):
                    error = "Please upload a CSV file."
                else:
                    # Process the CSV file
                    df = pd.read_csv(csv_file)
                    
                    # Get row count for display
                    row_count = len(df)
                    
                    # Store headers and data for display
                    headers = df.columns.tolist()
                    data = df.values.tolist()  # Show all rows
                    csv_data = True
                    
                    # Create visualizations
                    plots = create_visualizations(df)
                    
                    # Prepare data for ChartJS
                    features = headers[:-1]  # All columns except the last one
                    features_json = json.dumps(features)
                    
                    # Convert DataFrame to JSON for ChartJS
                    data_values = df.iloc[:, :-1].values.tolist()  # All rows, all columns except the last
                    data_json = json.dumps(data_values)
                    
                    # Get target classes for coloring
                    target_classes = df.iloc[:, -1].values.tolist()  # All rows, last column
                    target_classes_json = json.dumps(target_classes)
                    
                    # Get unique classes
                    unique_classes = sorted(df.iloc[:, -1].unique().tolist())
                    unique_classes_json = json.dumps(unique_classes)
                    
            except Exception as e:
                error = f"Error processing file: {str(e)}"
    else:
        form = CSVUploadForm()
    
    return render(request, 'project1/upload_csv.html', {
        'form': form,
        'error': error,
        'csv_data': csv_data,
        'headers': headers,
        'data': data,
        'plots': plots,
        'row_count': row_count,
        'filename': filename,
        'features': features,
        'features_json': features_json,
        'data_json': data_json,
        'target_classes': target_classes_json,
        'unique_classes': unique_classes_json,
    })


def create_visualizations(df):
    plots = []
    
    try:
        # Identify features and target
        features = df.iloc[:, :-1]  # All columns except the last one
        target = df.iloc[:, -1]     # Last column is the target
        
        # Get unique target values for coloring
        unique_targets = target.unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_targets)))
        
        # Create a scatter plot matrix for the first few features
        num_features = min(4, len(features.columns))
        
        # 1. Create a pair plot for the first few features
        plt.figure(figsize=(10, 8))
        for i in range(num_features):
            for j in range(num_features):
                plt.subplot(num_features, num_features, i*num_features + j + 1)
                
                if i == j:  # Diagonal: histogram
                    for t_idx, t_val in enumerate(unique_targets):
                        subset = features.iloc[:, i][target == t_val]
                        plt.hist(subset, alpha=0.5, color=colors[t_idx])
                    plt.xlabel(features.columns[i])
                else:  # Off-diagonal: scatter plot
                    for t_idx, t_val in enumerate(unique_targets):
                        mask = target == t_val
                        plt.scatter(
                            features.iloc[mask, j],
                            features.iloc[mask, i],
                            color=colors[t_idx],
                            alpha=0.5,
                            s=20
                        )
                    plt.xlabel(features.columns[j])
                    plt.ylabel(features.columns[i])
                
                plt.xticks([])
                plt.yticks([])
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plots.append(plot_data)
        
        # 2. Create a box plot for each feature
        plt.figure(figsize=(12, 6))
        features.boxplot()
        plt.title('Feature Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plots.append(plot_data)
        
        # 3. Target distribution
        plt.figure(figsize=(8, 6))
        target.value_counts().plot(kind='bar')
        plt.title('Target Distribution')
        plt.ylabel('Count')
        plt.xlabel('Class')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plots.append(plot_data)
        
    except Exception as e:
        # In case of error, return empty plots list
        print(f"Error creating visualizations: {str(e)}")
    
    return plots


# Create your views here.
