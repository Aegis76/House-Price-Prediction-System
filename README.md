```markdown
# 🏠 House Price Prediction System

End-to-end machine learning pipeline for predicting house prices (Boston Housing dataset).  
Includes data loading, cleaning, feature selection, model training (Linear/Ridge), hyperparameter tuning, evaluation, visualizations, and model persistence.

## ✨ Features

- **Automatic data loading** – from URL, local file, or synthetic fallback
- **Robust preprocessing** – duplicates removal, missing value imputation, constant feature dropping
- **Feature selection** – RFE (Recursive Feature Elimination) applied **only on training data** (no leakage)
- **Model training** – Linear Regression or Ridge Regression with optional grid search for `alpha`
- **Comprehensive evaluation** – R², Adjusted R², RMSE, MAE, cross‑validation, learning curves
- **Feature importance** – coefficient analysis
- **Visualizations** – actual vs predicted, residuals, coefficient bar plot, learning curve
- **Model persistence** – saves model, scaler, feature selector, and metadata
- **Inference function** – load a saved model and predict new house prices
- **Command‑line interface** – override configuration without editing code
- **Logging** – console + timestamped log file

## 📦 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

Install with:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic run (Ridge with auto hyperparameter tuning)
```bash
python house_price_pipeline.py
```

### Linear regression with custom test size
```bash
python house_price_pipeline.py --model-type linear --test-size 0.15
```

### Ridge with a fixed alpha (no tuning)
```bash
python house_price_pipeline.py --model-type ridge --alpha 2.0 --no-tune-alpha
```

### Load a previously saved model and run prediction demo
```bash
python house_price_pipeline.py --load-model house_price_model_20250614_120000.pkl --predict-demo
```

## 🧪 Output Files

| File | Description |
|------|-------------|
| `house_price_model_<timestamp>.pkl` | Pickled pipeline (model, scaler, feature selector, config) |
| `house_price_report_<timestamp>.txt` | Summary report with performance metrics |
| `model_analysis_<timestamp>.png` | Visualizations (actual vs predicted, residuals, coefficients, train/test R²) |
| `learning_curve_<timestamp>.png` | Learning curve to diagnose bias/variance |
| `pipeline_<timestamp>.log` | Detailed execution log |

## 🧠 Inference on a New House (programmatic)

```python
import pickle

# Load the saved pipeline
with open('house_price_model_20250614_120000.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Define feature values (must match the features used during training)
new_house = {
    'rm': 6.5,       # average number of rooms
    'lstat': 12.0,   # % lower status population
    'ptratio': 15.0, # pupil-teacher ratio
    'crim': 0.1,     # crime rate
    'nox': 0.5,      # nitric oxides concentration
    'chas': 0        # Charles River dummy
}

# Predict
from house_price_pipeline import predict_price  # or copy the function
predicted_price = predict_price(new_house, pipeline)
print(f"Predicted price: ${predicted_price * 1000:,.2f}")
```

## ⚙️ Command‑Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-type` | `linear` or `ridge` | `ridge` |
| `--alpha` | Regularization strength (Ridge) | `1.0` |
| `--tune-alpha` / `--no-tune-alpha` | Enable/disable alpha grid search | `True` |
| `--test-size` | Proportion of test set (0.0–1.0) | `0.2` |
| `--max-features` | Number of features to keep after RFE | `10` |
| `--cv-folds` | Number of cross‑validation folds | `5` |
| `--load-model` | Path to a saved `.pkl` model | `None` |
| `--predict-demo` | Run prediction demo using loaded model | `False` |

## 📁 Project Structure

```
.
├── house_price_pipeline.py      # Main script
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── .gitignore                   # Ignore generated files
├── tests/                       # (Optional) Unit tests
│   └── test_pipeline.py
└── config.yaml                  # (Optional) External configuration
```

## 🔧 Troubleshooting

### `matplotlib` plots not showing (headless environment)
The script automatically detects if `DISPLAY` is set and saves figures without showing them.  
To force display, install a GUI backend or run with `export DISPLAY=:0`.

### Missing `BostonHousing.csv`
The script will attempt to download from the URL. If that fails, it generates synthetic data with a realistic target–feature relationship.

### Feature selection error (not enough samples)
Increase `--max-features` or reduce `--cv-folds` to avoid over‑constraining RFE.


## 🤝 Contributing

Feel free to open issues or pull requests. Suggestions:
- Add more regression models (Lasso, ElasticNet, XGBoost)
- Deploy as a REST API (Flask/FastAPI)
- Add SHAP explanations for predictions

---

