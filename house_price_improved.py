import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import pickle
from typing import Tuple, List, Optional, Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION (with CLI overrides)
# ============================================================================

class Config:
    def __init__(self):
        self.data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        self.local_file = "BostonHousing.csv"
        self.target_column = 'medv'
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        self.min_samples = 20
        self.max_features = 10
        self.model_type = 'ridge'          # 'linear' or 'ridge'
        self.ridge_alpha = 1.0             # default, will be tuned if tune_alpha=True
        self.tune_alpha = True             # hyperparameter tuning for Ridge
        self.alpha_candidates = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        
        # File names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_file = f'house_price_model_{timestamp}.pkl'
        self.report_file = f'house_price_report_{timestamp}.txt'
        self.visualization_file = f'model_analysis_{timestamp}.png'
        self.learning_curve_file = f'learning_curve_{timestamp}.png'
        self.log_file = f'pipeline_{timestamp}.log'


CONFIG = Config()

# ============================================================================
# LOGGING (console + file)
# ============================================================================

def setup_logging(log_file: str):
    """Setup logging to both console and file"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = None  # will be set in main()

def print_section(title: str):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

# ============================================================================
# DATA LOADING (improved synthetic target)
# ============================================================================

def load_data() -> Optional[pd.DataFrame]:
    """Load data with fallback, error handling, and type normalization"""
    data = None
    
    # Try URL
    try:
        logger.info("Attempting to load data from URL...")
        data = pd.read_csv(CONFIG.data_url)
        logger.info(f"Successfully loaded from URL: {data.shape}")
    except Exception as e:
        logger.warning(f"Failed to load from URL: {e}")
    
    # Try local file
    if data is None:
        try:
            if os.path.exists(CONFIG.local_file):
                logger.info("Attempting to load from local file...")
                data = pd.read_csv(CONFIG.local_file)
                logger.info(f"Successfully loaded from local file: {data.shape}")
            else:
                logger.warning(f"Local file '{CONFIG.local_file}' not found")
        except Exception as e:
            logger.error(f"Failed to load from local file: {e}")
    
    # Generate synthetic data as last resort (with realistic relationship)
    if data is None:
        logger.info("Generating synthetic data with engineered target...")
        try:
            np.random.seed(CONFIG.random_state)
            n_samples = 506
            
            # Generate features with realistic distributions
            crim = np.random.exponential(3.6, n_samples)
            zn = np.random.choice([0, 20, 40, 60], n_samples)
            indus = np.random.normal(11, 6, n_samples)
            chas = np.random.binomial(1, 0.07, n_samples)
            nox = np.random.beta(2, 5, n_samples) * 0.9
            rm = np.random.normal(6.3, 0.7, n_samples)
            age = np.random.uniform(2, 100, n_samples)
            dis = np.random.exponential(3.8, n_samples)
            rad = np.random.choice(range(1, 10), n_samples)
            tax = np.random.normal(400, 160, n_samples)
            ptratio = np.random.normal(18, 2, n_samples)
            b = np.random.normal(360, 60, n_samples)
            lstat = np.random.beta(2, 4, n_samples) * 40
            
            # Build target as linear combination + noise (realistic)
            # Features with true effects: rm (+), lstat (-), ptratio (-), crim (-), nox (-)
            medv = (5.0 * rm 
                    - 0.5 * lstat 
                    - 0.3 * ptratio 
                    - 0.2 * crim 
                    - 10.0 * nox 
                    + 2.0 * chas 
                    + np.random.normal(0, 2, n_samples))
            medv = np.maximum(medv, 5)   # clip to realistic range
            
            data = pd.DataFrame({
                'crim': crim, 'zn': zn, 'indus': indus, 'chas': chas,
                'nox': nox, 'rm': rm, 'age': age, 'dis': dis,
                'rad': rad, 'tax': tax, 'ptratio': ptratio, 'b': b,
                'lstat': lstat, 'medv': medv
            })
            logger.info(f"Generated synthetic data: {data.shape}")
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            return None
    
    # Normalize known columns / types
    try:
        if 'chas' in data.columns:
            data['chas'] = pd.to_numeric(data['chas'], errors='coerce').fillna(0).astype(int)
    except Exception as e:
        logger.warning(f"Failed to normalize 'chas' column: {e}")
    
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Essential data cleaning with validation"""
    if data is None or data.empty:
        raise ValueError("Input data is empty or None")
    
    original_shape = data.shape
    logger.info(f"Original data shape: {original_shape}")
    
    # Drop duplicates
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Dropping {duplicates} duplicate rows")
        data = data.drop_duplicates()
    
    # Handle missing values
    missing_before = data.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"Found {missing_before} missing values")
        for col in data.columns:
            if data[col].isnull().any():
                if pd.api.types.is_numeric_dtype(data[col]):
                    fill_value = data[col].median()
                    data[col] = data[col].fillna(fill_value)
                else:
                    fill_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                    data[col] = data[col].fillna(fill_value)
    
    # Remove constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        logger.info(f"Removing constant columns: {constant_cols}")
        data = data.drop(columns=constant_cols)
    
    # Check for infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(data[col]).any():
            finite_vals = data[col][~np.isinf(data[col])]
            if not finite_vals.empty:
                data.loc[data[col] == np.inf, col] = finite_vals.max()
                data.loc[data[col] == -np.inf, col] = finite_vals.min()
    
    logger.info(f"Cleaned data shape: {data.shape}")
    return data

# ============================================================================
# FEATURE SELECTION (FIXED: only on training data)
# ============================================================================

def select_features(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, List[str], Any]:
    """
    Feature selection using RFE (fitted ONLY on training data).
    Returns selected training DataFrame, list of selected feature names, and the fitted selector.
    """
    if X_train.empty:
        raise ValueError("Feature matrix X_train is empty")
    
    if X_train.shape[1] <= CONFIG.max_features:
        logger.info(f"No feature selection needed. Using all {X_train.shape[1]} features")
        return X_train, X_train.columns.tolist(), None
    
    # Ensure no NaN values
    if X_train.isnull().any().any():
        logger.warning("Found NaN values in features. Imputing with column medians...")
        X_train = X_train.fillna(X_train.median())
    
    try:
        logger.info(f"Selecting {CONFIG.max_features} features from {X_train.shape[1]} using RFE...")
        
        if CONFIG.model_type == 'ridge':
            base_estimator = Ridge(alpha=CONFIG.ridge_alpha)
        else:
            base_estimator = LinearRegression()
        
        selector = RFE(base_estimator, n_features_to_select=CONFIG.max_features, step=1)
        selector.fit(X_train, y_train)
        
        support_mask = selector.get_support()
        selected_features = X_train.columns[support_mask].tolist()
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return X_train[selected_features], selected_features, selector
    except Exception as e:
        logger.error(f"Feature selection failed: {e}. Using all features.")
        return X_train, X_train.columns.tolist(), None


def apply_feature_selection(X: pd.DataFrame, selector: Any) -> pd.DataFrame:
    """Apply a fitted selector to a dataset (train or test)."""
    if selector is None:
        return X
    selected_features = X.columns[selector.get_support()].tolist()
    return X[selected_features]

# ============================================================================
# HYPERPARAMETER TUNING (Ridge alpha)
# ============================================================================

def tune_ridge_alpha(X_train: np.ndarray, y_train: pd.Series) -> float:
    """Grid search for best Ridge alpha using cross-validation."""
    logger.info("Tuning Ridge alpha using GridSearchCV...")
    param_grid = {'alpha': CONFIG.alpha_candidates}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=min(5, len(X_train)), 
                               scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    logger.info(f"Best alpha found: {best_alpha} (CV R² = {grid_search.best_score_:.4f})")
    return best_alpha


def train_model(X_train: np.ndarray, y_train: pd.Series) -> Ridge:
    """Train model with optional hyperparameter tuning."""
    if len(X_train) < CONFIG.min_samples:
        raise ValueError(f"Insufficient training samples: {len(X_train)}. Minimum: {CONFIG.min_samples}")
    
    if CONFIG.model_type == 'ridge':
        if CONFIG.tune_alpha:
            best_alpha = tune_ridge_alpha(X_train, y_train)
            CONFIG.ridge_alpha = best_alpha
        logger.info(f"Training Ridge regression with alpha={CONFIG.ridge_alpha}")
        model = Ridge(alpha=CONFIG.ridge_alpha)
    else:
        logger.info("Training Linear regression")
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

# ============================================================================
# EVALUATION, VISUALIZATION, REPORTING
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Comprehensive model evaluation with error handling"""
    print_section("MODEL EVALUATION")
    
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = {
            'r2': r2_score(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred)
        }
        
        test_metrics = {
            'r2': r2_score(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        n_test = len(y_test)
        p = len(feature_names)
        if n_test > p + 1:
            test_metrics['adj_r2'] = 1 - (1 - test_metrics['r2']) * (n_test - 1) / (n_test - p - 1)
        else:
            test_metrics['adj_r2'] = test_metrics['r2']
        
        # Cross-validation (simple KFold)
        n_splits = min(CONFIG.cv_folds, len(X_train))
        n_splits = max(2, n_splits)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=CONFIG.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        
        print(f"\nTraining Set (n={len(y_train)}):")
        print(f"  R²: {train_metrics['r2']:.4f}")
        print(f"  RMSE: ${train_metrics['rmse']*1000:,.2f}")
        print(f"  MAE: ${train_metrics['mae']*1000:,.2f}")
        
        print(f"\nTest Set (n={len(y_test)}):")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  Adj R²: {test_metrics['adj_r2']:.4f}")
        print(f"  RMSE: ${test_metrics['rmse']*1000:,.2f}")
        print(f"  MAE: ${test_metrics['mae']*1000:,.2f}")
        
        print(f"\nCross-Validation (R², {len(cv_scores)}-fold):")
        print(f"  Mean: {cv_scores.mean():.4f}")
        print(f"  Std:  {cv_scores.std():.4f}")
        
        r2_diff = train_metrics['r2'] - test_metrics['r2']
        if r2_diff > 0.15:
            print(f"\n⚠ WARNING: Potential overfitting (ΔR² = {r2_diff:.3f})")
        elif r2_diff > 0.1:
            print(f"\n⚠ Note: Moderate overfitting (ΔR² = {r2_diff:.3f})")
        else:
            print(f"\n✓ Good generalization (ΔR² = {r2_diff:.3f})")
        
        return y_train_pred, y_test_pred, train_metrics, test_metrics, cv_scores
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def analyze_features(model, feature_names):
    """Feature importance analysis"""
    print_section("FEATURE IMPORTANCE")
    
    if not hasattr(model, 'coef_'):
        print("Model does not have coefficients.")
        return None
    
    try:
        coef = model.coef_.flatten()
        if len(coef) != len(feature_names):
            n = min(len(coef), len(feature_names))
            coef = coef[:n]
            feature_names = feature_names[:n]
        
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef,
            'Abs_Impact': np.abs(coef)
        }).sort_values('Abs_Impact', ascending=False)
        
        print("\nTop Features:")
        print(importance.head(10).to_string(index=False))
        print(f"\nPositive coefficients: {sum(importance['Coefficient'] > 0)}")
        print(f"Negative coefficients: {sum(importance['Coefficient'] < 0)}")
        
        return importance
    except Exception as e:
        logger.error(f"Feature analysis failed: {e}")
        return None


def plot_learning_curve(model, X_train, y_train, cv_folds=5):
    """Plot learning curves to diagnose bias/variance."""
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv_folds, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('R² Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG.learning_curve_file, dpi=150)
    logger.info(f"Learning curve saved to {CONFIG.learning_curve_file}")
    if 'DISPLAY' in os.environ:
        plt.show()
    else:
        plt.close()


def create_visualizations(y_train, y_test, y_train_pred, y_test_pred, importance):
    """Create essential visualizations with headless support."""
    print_section("VISUALIZATIONS")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Actual vs Predicted (Test)
        axes[0,0].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', s=40)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Price ($1000s)')
        axes[0,0].set_ylabel('Predicted Price ($1000s)')
        axes[0,0].set_title('Actual vs Predicted (Test)')
        axes[0,0].grid(alpha=0.3)
        test_r2 = r2_score(y_test, y_test_pred)
        axes[0,0].text(0.05, 0.95, f'R² = {test_r2:.3f}', transform=axes[0,0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Residuals
        residuals = y_test - y_test_pred
        axes[0,1].scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', s=40)
        axes[0,1].axhline(0, color='r', linestyle='--', lw=2)
        axes[0,1].set_xlabel('Predicted Price ($1000s)')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residual Plot')
        axes[0,1].grid(alpha=0.3)
        
        # Feature importance
        if importance is not None and not importance.empty:
            top_n = min(8, len(importance))
            top_features = importance.head(top_n).reset_index(drop=True)
            colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
            axes[1,0].barh(range(top_n), top_features['Coefficient'], color=colors, alpha=0.7)
            axes[1,0].set_yticks(range(top_n))
            axes[1,0].set_yticklabels(top_features['Feature'])
            axes[1,0].set_xlabel('Coefficient')
            axes[1,0].set_title(f'Top {top_n} Coefficients')
            axes[1,0].axvline(0, color='black', linewidth=0.8)
        else:
            axes[1,0].text(0.5, 0.5, 'Feature Importance\nNot Available', ha='center', va='center')
            axes[1,0].set_title('Feature Importance')
        
        # Train vs Test R²
        train_r2 = r2_score(y_train, y_train_pred)
        bars = axes[1,1].bar(['Train R²', 'Test R²'], [train_r2, test_r2], alpha=0.7, width=0.6)
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_title('Train vs Test Performance')
        axes[1,1].grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, [train_r2, test_r2]):
            axes[1,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{val:.3f}', ha='center', va='bottom')
        
        r2_diff = train_r2 - test_r2
        color = 'red' if r2_diff > 0.15 else 'orange' if r2_diff > 0.1 else 'green'
        axes[1,1].text(0.95, 0.05, f'ΔR² = {r2_diff:.3f}', transform=axes[1,1].transAxes,
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3), ha='right')
        
        plt.tight_layout()
        plt.savefig(CONFIG.visualization_file, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved as '{CONFIG.visualization_file}'")
        if 'DISPLAY' in os.environ:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def generate_report(data_shape, selected_features, test_metrics, cv_scores, model_type, importance=None):
    """Generate concise report."""
    print_section("REPORT GENERATION")
    
    try:
        report = [
            "HOUSE PRICE PREDICTION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATA SUMMARY",
            "-" * 30,
            f"Samples: {data_shape[0]:,}",
            f"Features: {data_shape[1]:,}",
            f"Features selected: {len(selected_features):,}",
            "",
            "MODEL CONFIGURATION",
            "-" * 30,
            f"Algorithm: {model_type.upper()}",
            f"Ridge alpha: {CONFIG.ridge_alpha if model_type == 'ridge' else 'N/A'}",
            f"Hyperparameter tuning: {CONFIG.tune_alpha}",
            f"Max features: {CONFIG.max_features}",
            "",
            "MODEL PERFORMANCE",
            "-" * 30,
            f"Test R²: {test_metrics.get('r2', 0):.4f}",
            f"Adjusted R²: {test_metrics.get('adj_r2', test_metrics.get('r2', 0)):.4f}",
            f"RMSE: ${test_metrics.get('rmse', 0)*1000:,.2f}",
            f"MAE: ${test_metrics.get('mae', 0)*1000:,.2f}",
            "",
            "CROSS-VALIDATION",
            "-" * 30,
            f"Folds: {len(cv_scores)}",
            f"Mean R²: {cv_scores.mean():.4f}",
            f"Std Dev: {cv_scores.std():.4f}",
            "",
        ]
        
        if importance is not None and not importance.empty:
            report.append("TOP 5 FEATURES")
            report.append("-" * 30)
            for i, (_, row) in enumerate(importance.head(5).iterrows(), 1):
                direction = "Increases" if row['Coefficient'] > 0 else "Decreases"
                report.append(f"{i}. {row['Feature']}: {direction} price (coef: {row['Coefficient']:.4f})")
            report.append("")
        
        report.append("INTERPRETATION")
        report.append("-" * 30)
        r2 = test_metrics.get('r2', 0)
        if r2 >= 0.8:
            report.append("✓ Excellent model performance")
        elif r2 >= 0.6:
            report.append("✓ Good model performance")
        elif r2 >= 0.4:
            report.append("⚠ Moderate model performance")
        else:
            report.append("✗ Poor model performance")
        
        report.append("=" * 50)
        
        with open(CONFIG.report_file, 'w') as f:
            f.write('\n'.join(report))
        print(f"✓ Report saved to {CONFIG.report_file}")
        return True
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False


def save_model(model, scaler, feature_names, selector, metrics):
    """Save model, scaler, feature selector, and metadata."""
    try:
        pipeline = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'feature_selector': selector,   # fitted RFE selector
            'metrics': metrics,
            'config': {
                'model_type': CONFIG.model_type,
                'ridge_alpha': CONFIG.ridge_alpha if CONFIG.model_type == 'ridge' else None,
                'max_features': CONFIG.max_features,
                'tune_alpha': CONFIG.tune_alpha,
                'timestamp': datetime.now().isoformat()
            }
        }
        with open(CONFIG.model_file, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"\n✓ Model saved to {CONFIG.model_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False


def load_model(model_path: str) -> Dict[str, Any]:
    """Load a saved model pipeline."""
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return pipeline


def predict_price(features_dict: Dict[str, float], model_pipeline: Dict[str, Any]) -> float:
    """
    Predict house price for a single instance.
    features_dict: dictionary with feature names and values (must match training features).
    model_pipeline: output from load_model().
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([features_dict])
    # Ensure only the features used during training
    expected_features = model_pipeline['feature_names']
    for col in expected_features:
        if col not in input_df.columns:
            raise ValueError(f"Missing feature: {col}")
    input_df = input_df[expected_features]
    
    # Apply feature selector (if exists)
    if model_pipeline.get('feature_selector') is not None:
        input_df = apply_feature_selection(input_df, model_pipeline['feature_selector'])
    
    # Scale
    X_scaled = model_pipeline['scaler'].transform(input_df)
    # Predict
    prediction = model_pipeline['model'].predict(X_scaled)[0]
    return prediction

# ============================================================================
# MAIN PIPELINE WITH CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="House Price Prediction Pipeline")
    parser.add_argument('--model-type', choices=['linear', 'ridge'], default='ridge',
                        help='Regression algorithm')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge regularization strength (if tuning disabled)')
    parser.add_argument('--tune-alpha', action='store_true', default=True,
                        help='Perform hyperparameter tuning for Ridge alpha')
    parser.add_argument('--no-tune-alpha', dest='tune_alpha', action='store_false',
                        help='Disable hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of test set')
    parser.add_argument('--max-features', type=int, default=10,
                        help='Number of features to select via RFE')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to saved model for inference demo')
    parser.add_argument('--predict-demo', action='store_true',
                        help='Run a prediction demo using a saved model')
    return parser.parse_args()


def main():
    global logger, CONFIG
    
    # Parse CLI arguments
    args = parse_args()
    CONFIG.model_type = args.model_type
    CONFIG.ridge_alpha = args.alpha
    CONFIG.tune_alpha = args.tune_alpha
    CONFIG.test_size = args.test_size
    CONFIG.max_features = args.max_features
    CONFIG.cv_folds = args.cv_folds
    
    # Setup logging with timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CONFIG.log_file = f'pipeline_{timestamp}.log'
    logger = setup_logging(CONFIG.log_file)
    
    logger.info("House Price Prediction Pipeline started")
    logger.info(f"Configuration: {vars(CONFIG)}")
    
    # If loading model for demo
    if args.load_model and args.predict_demo:
        print_section("PREDICTION DEMO USING SAVED MODEL")
        try:
            model_pipeline = load_model(args.load_model)
            # Example: use median values from training? We don't have them saved, so use dummy.
            sample_features = {feat: 0.0 for feat in model_pipeline['feature_names']}
            # Fill with some realistic values (hardcoded for Boston)
            if 'rm' in sample_features: sample_features['rm'] = 6.5
            if 'lstat' in sample_features: sample_features['lstat'] = 12.0
            if 'ptratio' in sample_features: sample_features['ptratio'] = 15.0
            pred_price = predict_price(sample_features, model_pipeline)
            print(f"\nPredicted price for example house: ${pred_price*1000:,.2f}")
            print("Demo completed.")
        except Exception as e:
            print(f"Demo failed: {e}")
        return
    
    print_section("HOUSE PRICE PREDICTION PIPELINE")
    
    try:
        # 1. Load data
        print("\n1. Loading data...")
        data = load_data()
        if data is None or data.empty:
            print("✗ Failed to load data. Exiting.")
            return
        print(f"   ✓ Loaded {data.shape[0]} samples, {data.shape[1]} features")
        
        # 2. Clean data
        print("\n2. Cleaning data...")
        data_clean = clean_data(data)
        print(f"   ✓ Cleaned shape: {data_clean.shape}")
        
        # 3. Prepare features
        if CONFIG.target_column not in data_clean.columns:
            print(f"✗ Target column '{CONFIG.target_column}' not found")
            return
        
        X = data_clean.drop(columns=[CONFIG.target_column])
        y = data_clean[CONFIG.target_column]
        print(f"   ✓ Target: '{CONFIG.target_column}'")
        
        # 4. Split data (BEFORE any feature selection or scaling)
        print("\n3. Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG.test_size, random_state=CONFIG.random_state, shuffle=True
        )
        print(f"   ✓ Train: {X_train.shape[0]} samples")
        print(f"   ✓ Test:  {X_test.shape[0]} samples")
        
        # 5. Feature selection (ONLY on training data)
        print("\n4. Selecting features (using training data only)...")
        X_train_selected, selected_features, selector = select_features(X_train, y_train)
        X_test_selected = apply_feature_selection(X_test, selector)
        print(f"   ✓ Selected {len(selected_features)} features")
        
        # 6. Scale features
        print("\n5. Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        print(f"   ✓ Features scaled")
        
        # 7. Train model (with optional hyperparameter tuning)
        print("\n6. Training model...")
        model = train_model(X_train_scaled, y_train)
        print(f"   ✓ Model: {CONFIG.model_type.upper()}")
        
        # 8. Evaluate
        print("\n7. Evaluating model...")
        y_train_pred, y_test_pred, train_metrics, test_metrics, cv_scores = evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, selected_features
        )
        test_metrics['train_r2'] = train_metrics['r2']
        
        # 9. Feature analysis
        importance = analyze_features(model, selected_features)
        
        # 10. Learning curve (optional, but useful)
        print("\n8. Generating learning curve...")
        try:
            plot_learning_curve(model, X_train_scaled, y_train, cv_folds=min(5, len(X_train)))
        except Exception as e:
            logger.warning(f"Learning curve skipped: {e}")
        
        # 11. Visualizations
        print("\n9. Generating visualizations...")
        create_visualizations(y_train, y_test, y_train_pred, y_test_pred, importance)
        
        # 12. Save model
        print("\n10. Saving model...")
        save_model(model, scaler, selected_features, selector, {
            'train': train_metrics,
            'test': test_metrics,
            'cv': {'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores.tolist()}
        })
        
        # 13. Generate report
        generate_report(data_clean.shape, selected_features, test_metrics, cv_scores, 
                        CONFIG.model_type, importance)
        
        # 14. Sample predictions using only training data median
        print_section("SAMPLE PREDICTIONS")
        # Use median from training set (safe)
        sample_median = X_train_selected.median().values.reshape(1, -1)
        sample_scaled = scaler.transform(sample_median)
        prediction = model.predict(sample_scaled)[0]
        print(f"Predicted price for median training house: ${prediction*1000:,.2f}")
        
        # Random test sample
        if len(X_test) > 0:
            idx = np.random.randint(0, len(X_test))
            x_sample = X_test_selected.iloc[idx:idx+1].values
            x_sample_scaled = scaler.transform(x_sample)
            y_pred_sample = model.predict(x_sample_scaled)[0]
            actual_price = y_test.iloc[idx]
            error_abs = abs(actual_price - y_pred_sample)
            error_pct = error_abs / actual_price * 100 if actual_price != 0 else float('inf')
            print(f"\nRandom test sample #{idx}:")
            print(f"  Actual:    ${actual_price*1000:,.2f}")
            print(f"  Predicted: ${y_pred_sample*1000:,.2f}")
            print(f"  Error:     ${error_abs*1000:,.2f} ({error_pct:.1f}%)")
        
        print_section("ANALYSIS COMPLETE")
        print(f"✓ Model saved: {CONFIG.model_file}")
        print(f"✓ Report saved: {CONFIG.report_file}")
        print(f"✓ Visualization: {CONFIG.visualization_file}")
        print(f"✓ Learning curve: {CONFIG.learning_curve_file}")
        print(f"✓ Log file: {CONFIG.log_file}")
        
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user.")
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()
