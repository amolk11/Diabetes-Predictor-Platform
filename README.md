# Diabetes Prediction ML Project

Production-ready machine learning system for diabetes prediction with 25+ evaluation metrics.

## Features

✅ 7 ML Algorithms (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM)
✅ 25+ Evaluation Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, etc.)
✅ Cross-validation Support
✅ MLflow Integration
✅ DVC Data Versioning
✅ FastAPI REST Server
✅ Docker Support
✅ AWS Deployment Ready

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Train Models
python scripts/train.py

# Evaluate
python scripts/evaluate.py
```

## Project Structure

```
.
├── src/                 # Source code package
│   ├── data/           # Data loading & preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Model implementations
│   ├── training/       # Training pipeline
│   ├── inference/      # Prediction module
│   └── utils/          # Utilities
├── scripts/            # Executable scripts
│   ├── train.py       # Training pipeline
│   ├── evaluate.py    # Evaluation script
│   └── deploy.py      # AWS deployment
├── config/            # Configuration files
├── models/            # Trained models
├── data/              # Dataset directory
└── logs/              # Application logs
```

## Models Included

1. **Logistic Regression** - Fast baseline
2. **Random Forest** - Ensemble method
3. **Gradient Boosting** - Powerful boosting
4. **XGBoost** - Optimized gradient boosting
5. **LightGBM** - Fast gradient boosting
6. **CatBoost** - Categorical features
7. **SVM** - Support vector machine

## Evaluation Metrics

### Basic Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Advanced Metrics
- Matthews Correlation Coefficient
- Cohen's Kappa
- Confusion Matrix
- Classification Report
- Cross-validation Scores

## Next Steps

1. Place your diabetes.csv in data/ folder
2. Run `python scripts/train.py`
3. Check results in models/ folder
4. Deploy using `python scripts/deploy.py`

## Requirements

- Python 3.8+
- 4GB+ RAM
- 10GB+ Disk Space

## License

MIT

## Author

Data Science Team
