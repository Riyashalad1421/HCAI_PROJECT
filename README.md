# Human-Centric Artificial Intelligence - Project Portfolio

**Course**: Human-Centric Artificial Intelligence  
**Academic Year**: 2024-2025  
**Framework**: Django Web Applications

## Repository Overview

This repository contains five comprehensive Django applications demonstrating various aspects of human-centric AI, from automated machine learning interfaces to reinforcement learning with human feedback. Each project emphasizes user interaction, transparency, and human-centered design principles in AI systems.

## Repository Structure

```
HCAI-PBL/
├── demos/                         # Demo applications and examples
├── home/                          # Landing page and navigation
├── media/                         # User-uploaded files and generated content
├── pbl/                           # Main Django project configuration
├── project1/                      # Automated Machine Learning Interface
├── project2/                      # Active Learning for Text Classification
├── project3/                      # Explainability and Interpretability
├── project4/                      # Recommender Systems with User Study
├── project5/                      # Reinforcement Learning with Human Feedback
├── static/                        # Global CSS, JavaScript, and assets
├── temp_sessions/                 # Temporary session data
├── templates/                     # Shared HTML templates
├── .gitignore                     # Git ignore configuration
├── db.sqlite3                     # SQLite database (auto-generated)
├── manage.py                      # Django management script
├── README.md                      # This documentation
└── requirements.txt               # All project dependencies
```

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git
- Modern web browser

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd HCAI-PBL
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install all dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required datasets** (see individual project sections below)

5. **Setup Django database**
```bash
python manage.py makemigrations
python manage.py migrate
```

6. **Run the development server**
```bash
python manage.py runserver
```

7. **Access the application**
Navigate to `http://127.0.0.1:8000/home/` in your web browser.

## Project Descriptions

### Project 1: Automated Machine Learning Interface
**URL**: `/project1/`

An end-to-end supervised learning interface allowing users to:
- Upload CSV datasets 
- Visualize data with interactive plots
- Train multiple ML models (classification/regression)
- Compare model performance with hyperparameter tuning
- Export results and trained models

**Key Technologies**: scikit-learn, pandas, matplotlib, seaborn
**Datasets**: User-uploaded CSV files (Iris, Wine, Boston Housing examples provided)

### Project 2: Active Learning for Text Classification
**URL**: `/project2/`

Interactive text classification system implementing active learning strategies:
- Sentiment analysis on IMDB movie reviews (50k dataset)
- Pool-based active learning with multiple utility functions
- Real-time model training with user feedback
- Batch and stream-based active learning extensions
- Performance comparison with passive learning

**Key Technologies**: scikit-learn, NLTK/spaCy, TF-IDF, word embeddings
**Dataset**: IMDB 50k Movie Reviews (download required)

### Project 3: Explainability and Interpretability
**URL**: `/project3/`

Explainable AI interface using Palmer Penguins dataset:
- Decision tree visualization with complexity control
- Logistic regression with feature importance
- Interactive sparsity sliders (λ parameter control)
- Counterfactual explanation generation
- Model complexity vs. accuracy trade-offs

**Key Technologies**: scikit-learn, GOSDT, visualization libraries
**Dataset**: Palmer Penguins (via palmerpenguins package)

### Project 4: Recommender Systems with User Study
**URL**: `/project4/`

Transparent recommendation system with user study interface:
- MovieLens-based movie recommendations
- Matrix factorization with rating impact visualization
- Interactive user study for cold-start scenarios
- Real-time analytics dashboard
- Comprehensive feedback collection system

**Key Technologies**: pandas, numpy, Chart.js, matrix factorization
**Dataset**: MovieLens 100k (9k movies, 600 users)

### Project 5: Reinforcement Learning with Human Feedback
**URL**: `/project5/`

Grid-world game with human feedback integration:
- 5x5 mouse and cheese environment
- REINFORCE algorithm implementation
- Bradley-Terry preference model
- Interactive policy refinement interface
- Human feedback collection and integration

**Key Technologies**: PyTorch, reinforcement learning algorithms
**Environment**: Custom grid-world implementation

## Installation Details by Project

### Project 1 Dependencies
```bash
# Core ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Project 2 Dependencies
```bash
# NLP and text processing
pip install scikit-learn nltk spacy pandas numpy
pip install datasets  # For IMDB dataset
python -m spacy download en_core_web_sm
```

### Project 3 Dependencies
```bash
# Decision trees and interpretability
pip install scikit-learn palmerpenguins pandas numpy
pip install gosdt  # For sparse decision trees
```

### Project 4 Dependencies
```bash
# Recommender systems
pip install pandas numpy scikit-learn reportlab
```

### Project 5 Dependencies
```bash
# Deep learning and RL
pip install torch torchvision numpy matplotlib
```

## Dataset Setup

### Required Datasets

1. **Project 2: IMDB Reviews**
   - Download from: https://ai.stanford.edu/~amaas/data/sentiment/
   - Place in: `project2/data/imdb_reviews/`

2. **Project 3: Palmer Penguins**
   - Automatically downloaded via palmerpenguins package
   - No manual setup required

3. **Project 4: MovieLens 100k**
   - Download from: https://grouplens.org/datasets/movielens/
   - Place `movies.csv` and `ratings.csv` in: `project4/data/`

4. **Project 5: Grid Environment**
   - Custom implementation included
   - No external dataset required

## Usage Guide

### For Students/Developers

1. **Start with the home page** (`/home/`) to understand the overall structure
2. **Navigate to individual projects** using the provided links
3. **Follow project-specific instructions** in each app's documentation
4. **Check console logs** for debugging information during development

### For Instructors/Evaluators

1. **Access the main navigation** at `/home/`
2. **Each project is self-contained** with its own interface and documentation
3. **All required dependencies** are listed in requirements.txt
4. **Database migrations** handle any required data models automatically

## Development Notes

### Django App Structure
Each project follows standard Django patterns:
- `models.py`: Data models and database schema
- `views.py`: Request handling and business logic  
- `urls.py`: URL routing and patterns
- `templates/`: HTML templates with Django templating
- `static/`: CSS, JavaScript, and image assets

### Key Features Across Projects
- **Responsive Design**: All interfaces work on desktop and mobile
- **Interactive Visualizations**: Real-time charts and plots
- **User Feedback Systems**: Data collection for human-centric evaluation
- **Export Functionality**: Download results and reports
- **Error Handling**: Graceful handling of edge cases and invalid inputs

## Common Issues and Solutions

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Database Issues
```bash
# Reset database (WARNING: deletes all data)
rm db.sqlite3
python manage.py makemigrations
python manage.py migrate
```

### Dataset Loading Problems
- Verify dataset files are in correct locations
- Check file formats match expected CSV structure
- Ensure sufficient disk space for large datasets

### Performance Issues
- Large datasets may require longer processing times
- Consider using smaller dataset samples during development
- Monitor memory usage with memory-intensive operations

## Extending the Projects

### Adding New Features
1. Follow Django best practices for new views and models
2. Update requirements.txt for any new dependencies
3. Add appropriate error handling and user feedback
4. Include responsive design for mobile compatibility

### Customizing Interfaces
1. Modify templates in respective `templates/` directories
2. Update CSS in project-specific `static/` folders
3. Test across different screen sizes and browsers
4. Maintain accessibility standards

## Academic Context

### Learning Objectives
- **Human-Computer Interaction**: Design interfaces that enhance user understanding of AI
- **Machine Learning Pipeline**: Implement complete ML workflows with user interaction
- **Explainable AI**: Create transparent and interpretable AI systems
- **User Studies**: Design and implement human-centered evaluation methodologies
- **Ethical AI**: Consider human factors in AI system design and deployment

### Evaluation Criteria
- **Functionality**: All features work as specified
- **User Experience**: Intuitive and responsive interfaces
- **Code Quality**: Clean, documented, and maintainable code
- **Innovation**: Creative approaches to human-centric AI challenges
- **Documentation**: Clear instructions and explanations

## Technical Specifications

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for datasets and models
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

### Performance Benchmarks
- **Project 1**: Handles datasets up to 100k rows
- **Project 2**: IMDB 50k reviews (~500MB memory usage)
- **Project 3**: Real-time decision tree updates (<1s response)
- **Project 4**: 9k movies, 600 users recommendation matrix
- **Project 5**: 60+ FPS grid-world visualization

## Contributing

### Code Standards
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Add comments for complex algorithms or business logic

### Testing
- Test all user workflows manually
- Verify responsive design on multiple devices
- Check error handling with invalid inputs
- Validate data export/import functionality

## License and Usage

This project is developed for academic purposes as part of the Human-Centric Artificial Intelligence course. All code and documentation are provided for educational use.

## Support and Contact

For technical issues or questions about the implementation, refer to:
1. Individual project documentation in respective directories
2. Django official documentation
3. Library-specific documentation for scikit-learn, PyTorch, etc.
4. Course materials and lecture notes

---

**Note**: This is a comprehensive academic project demonstrating multiple aspects of human-centric AI. Each component is designed to be both educational and practically applicable to real-world AI system development.
