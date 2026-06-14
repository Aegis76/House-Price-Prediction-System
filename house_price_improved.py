
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import pickle
from typing import Tuple, List, Optional

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Simplified configuration"""
    def __init__(self):
        self.data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        self.local_file = "BostonHousing.csv"
        self.target_column = 'medv'
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        self.min_samples = 20
        self.max_features = 10
        self.model_type = 'ridge'  # 'linear' or 'ridge'
        self.ridge_alpha = 1.0
        
        # File names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_file = f'house_price_model_{timestamp}.pkl'
        self.report_file = f'house_price_report_{timestamp}.txt'
        self.visualization_file = f'model_analysis_{timestamp}.png'


CONFIG = Config()

# ============================================================================
# ESSENTIAL HELPER FUNCTIONS
# ============================================================================

def setup_logging():
    """Minimal logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def print_section(title: str):
    """Simple section separator"""
    print(f"\n{'='*60}")
    print(title)
    print('='*60)


def load_data() -> Optional[pd.DataFrame]:
    """Load data with fallback, error handling, and type normalization"""
    data = None
    
    # Try URL
    try:
        logger.info("Attempting to load data from URL...")
        data = pd.read_csv(CONFIG.data_url)
        logger.info(f"Successfully loaded from URL: {data.shape[0]} samples, {data.shape[1]} features")
    except Exception as e:
        logger.warning(f"Failed to load from URL: {e}")
    
    # Try local file
    if data is None:
        try:
            if os.path.exists(CONFIG.local_file):
                logger.info("Attempting to load from local file...")
                data = pd.read_csv(CONFIG.local_file)
                logger.info(f"Successfully loaded from local file: {data.shape[0]} samples, {data.shape[1]} features")
            else:
                logger.warning(f"Local file '{CONFIG.local_file}' not found")
        except Exception as e:
            logger.error(f"Failed to load from local file: {e}")
            data = None
    
    # Generate synthetic data as last resort
    if data is None:
        logger.info("Generating synthetic data...")
        try:
            np.random.seed(CONFIG.random_state)
            n_samples = 506
            data = pd.DataFrame({
                'crim': np.random.exponential(3.6, n_samples),
                'zn': np.random.choice([0, 20, 40, 60], n_samples),
                'indus': np.random.normal(11, 6, n_samples),
                'chas': np.random.binomial(1, 0.07, n_samples),
                'nox': np.random.beta(2, 5, n_samples) * 0.9,
                'rm': np.random.normal(6.3, 0.7, n_samples),
                'age': np.random.uniform(2, 100, n_samples),
                'dis': np.random.exponential(3.8, n_samples),
                'rad': np.random.choice(range(1, 10), n_samples),
                'tax': np.random.normal(400, 160, n_samples),
                'ptratio': np.random.normal(18, 2, n_samples),
                'b': np.random.normal(360, 60, n_samples),
                'lstat': np.random.beta(2, 4, n_samples) * 40,
                'medv': np.random.normal(22, 9, n_samples)
            })
            logger.info(f"Generated synthetic data: {data.shape[0]} samples, {data.shape[1]} features")
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            return None
    
    # Normalize known columns / types
    try:
        if 'chas' in data.columns:
            # Safer numeric conversion for 'chas'
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
                    logger.debug(f"Filled missing values in '{col}' with median: {fill_value:.4f}")
                else:
                    fill_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                    data[col] = data[col].fillna(fill_value)
                    logger.debug(f"Filled missing values in '{col}' with mode: {fill_value}")
    
    # Remove constant columns
    constant_cols = []
    for col in data.columns:
        if data[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        data = data.drop(columns=constant_cols)
    
    # Check for infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(data[col]).any():
            logger.warning(f"Found infinite values in '{col}', replacing with max/min")
            finite_vals = data[col][~np.isinf(data[col])]
            if not finite_vals.empty:
                data.loc[data[col] == np.inf, col] = finite_vals.max()
                data.loc[data[col] == -np.inf, col] = finite_vals.min()
    
    logger.info(f"Cleaned data shape: {data.shape}")
    return data


def select_features(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
    """Feature selection using RFE WITHOUT scaling (scaling done later)"""
    if X.empty:
        raise ValueError("Feature matrix X is empty")
    
    if X.shape[1] <= CONFIG.max_features:
        logger.info(f"No feature selection needed. Using all {X.shape[1]} features")
        return X, X.columns.tolist()
    
    # Ensure no NaN values for RFE
    if X.isnull().any().any():
        logger.warning("Found NaN values in features. Imputing with column medians...")
        X = X.fillna(X.median())
    
    try:
        logger.info(f"Selecting {CONFIG.max_features} features from {X.shape[1]} using RFE...")
        
        if CONFIG.model_type == 'ridge':
            base_estimator = Ridge(alpha=CONFIG.ridge_alpha)
        else:
            base_estimator = LinearRegression()
        
        selector = RFE(base_estimator, n_features_to_select=CONFIG.max_features, step=1)
        selector.fit(X, y)
        
        support_mask = selector.get_support()
        selected_features = X.columns[support_mask].tolist()
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return X[selected_features], selected_features
    except Exception as e:
        logger.error(f"Feature selection failed: {e}. Using all features.")
        return X, X.columns.tolist()


def train_model(X_train: np.ndarray, y_train: pd.Series):
    """Train model with selected algorithm"""
    if len(X_train) < CONFIG.min_samples:
        raise ValueError(f"Insufficient training samples: {len(X_train)}. Minimum required: {CONFIG.min_samples}")
    
    if CONFIG.model_type == 'ridge':
        logger.info(f"Training Ridge regression with alpha={CONFIG.ridge_alpha}")
        model = Ridge(alpha=CONFIG.ridge_alpha)
    else:
        logger.info("Training Linear regression")
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Comprehensive model evaluation with error handling"""
    print_section("MODEL EVALUATION")
    
    try:
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
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
        
        # Adjusted R-squared
        n_test = len(y_test)
        p = len(feature_names)
        if n_test > p + 1:
            test_metrics['adj_r2'] = 1 - (1 - test_metrics['r2']) * (n_test - 1) / (n_test - p - 1)
        else:
            test_metrics['adj_r2'] = test_metrics['r2']
        
        # Cross-validation with KFold
        max_cv = CONFIG.cv_folds
        # ensure n_splits is sensible: at least 2 and at most len(X_train)
        n_splits = min(max_cv, max(2, min(len(X_train), max(2, len(X_train) // 5))))
        n_splits = max(2, min(n_splits, len(X_train)))
        logger.info(f"Using KFold cross-validation with n_splits={n_splits}")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=CONFIG.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        
        # Print results
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
        print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        
        # Overfitting check
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
    """Feature importance analysis with error handling"""
    print_section("FEATURE IMPORTANCE")
    
    if not hasattr(model, 'coef_'):
        print("Model does not have coefficients for feature importance analysis")
        return None
    
    try:
        # Handle different coefficient formats
        if isinstance(model.coef_, np.ndarray):
            coef = model.coef_.flatten()
        else:
            coef = np.array(model.coef_).flatten()
        
        if len(coef) != len(feature_names):
            logger.warning(f"Coef length ({len(coef)}) doesn't match feature count ({len(feature_names)})")
            # Use available coefficients
            n = min(len(coef), len(feature_names))
            coef = coef[:n]
            feature_names = feature_names[:n]
        
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef,
            'Abs_Impact': np.abs(coef)
        }).sort_values('Abs_Impact', ascending=False)
        
        print("\nTop Features:")
        print(importance.head(min(10, len(importance))).to_string(index=False))
        
        # Summary statistics
        print(f"\nFeature Analysis:")
        print(f"  Positive coefficients: {sum(importance['Coefficient'] > 0)}")
        print(f"  Negative coefficients: {sum(importance['Coefficient'] < 0)}")
        print(f"  Mean |coefficient|: {importance['Abs_Impact'].mean():.4f}")
        
        return importance
        
    except Exception as e:
        logger.error(f"Feature analysis failed: {e}")
        return None


def create_visualizations(y_train, y_test, y_train_pred, y_test_pred, importance):
    """Create essential visualizations with error handling"""
    print_section("VISUALIZATIONS")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Actual vs Predicted (Test)
        axes[0,0].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', s=40)
        axes[0,0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
        axes[0,0].set_xlabel('Actual Price ($1000s)')
        axes[0,0].set_ylabel('Predicted Price ($1000s)')
        axes[0,0].set_title('Actual vs Predicted (Test Set)')
        axes[0,0].grid(alpha=0.3)
        axes[0,0].legend()
        
        # Add R² annotation
        test_r2 = r2_score(y_test, y_test_pred)
        axes[0,0].text(0.05, 0.95, f'R² = {test_r2:.3f}', 
                       transform=axes[0,0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Residuals plot
        residuals = y_test - y_test_pred
        axes[0,1].scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', s=40)
        axes[0,1].axhline(0, color='r', linestyle='--', lw=2)
        axes[0,1].set_xlabel('Predicted Price ($1000s)')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residual Plot')
        axes[0,1].grid(alpha=0.3)
        
        # Add residual statistics
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        axes[0,1].axhline(residual_mean, color='g', linestyle=':', lw=1, label=f'Mean: {residual_mean:.2f}')
        axes[0,1].legend()
        
        # 3. Feature importance
        if importance is not None and not importance.empty:
            top_n = min(8, len(importance))
            top_features = importance.head(top_n).reset_index(drop=True)
            colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
            
            axes[1,0].barh(range(top_n), top_features['Coefficient'].values,
                          color=colors, edgecolor='k', alpha=0.7)
            axes[1,0].set_yticks(range(top_n))
            axes[1,0].set_yticklabels(top_features['Feature'].values)
            axes[1,0].set_xlabel('Coefficient Value')
            axes[1,0].set_title(f'Top {top_n} Feature Coefficients')
            axes[1,0].axvline(0, color='black', linewidth=0.8)
        else:
            axes[1,0].text(0.5, 0.5, 'Feature Importance\nNot Available',
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Feature Importance')
        
        # 4. Performance comparison
        train_r2 = r2_score(y_train, y_train_pred)
        bars = axes[1,1].bar(['Train R²', 'Test R²'],
                            [train_r2, test_r2],
                            alpha=0.7, width=0.6)
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_title('Train vs Test Performance')
        axes[1,1].grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, [train_r2, test_r2]):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{val:.3f}', ha='center', va='bottom')
        
        # Add overfitting indicator
        r2_diff = train_r2 - test_r2
        if r2_diff > 0.15:
            color = 'red'
            text = f'Overfitting: {r2_diff:.3f}'
        elif r2_diff > 0.1:
            color = 'orange'
            text = f'Moderate: {r2_diff:.3f}'
        else:
            color = 'green'
            text = f'Good: {r2_diff:.3f}'
        
        axes[1,1].text(0.95, 0.05, text,
                       transform=axes[1,1].transAxes,
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                       ha='right', va='bottom')
        
        plt.tight_layout()
        try:
            plt.savefig(CONFIG.visualization_file, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved as '{CONFIG.visualization_file}'")
        except Exception as e:
            print(f"✗ Could not save visualization: {e}")
        # Show only when running interactively
        try:
            plt.show()
        except Exception:
            pass
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        print(f"Visualization failed: {e}")


def save_model(model, scaler, feature_names, metrics):
    """Save trained model with error handling"""
    try:
        pipeline = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'metrics': metrics,
            'config': {
                'model_type': CONFIG.model_type,
                'ridge_alpha': CONFIG.ridge_alpha if CONFIG.model_type == 'ridge' else None,
                'max_features': CONFIG.max_features,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(CONFIG.model_file, 'wb') as f:
            pickle.dump(pipeline, f)
        
        print(f"\n✓ Model saved to {CONFIG.model_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        print(f"✗ Failed to save model: {e}")
        return False


def generate_report(data_shape, selected_features, test_metrics, cv_scores, model_type, importance=None):
    """Generate concise report with proper formatting"""
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
            f"Algorithm: {model_type.upper()} Regression",
            f"Ridge alpha: {CONFIG.ridge_alpha if model_type == 'ridge' else 'N/A'}",
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
            f"Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]",
            "",
        ]
        
        # Add top features if available
        if importance is not None and not importance.empty:
            report.append("TOP 5 FEATURES")
            report.append("-" * 30)
            top_features = importance.head(5)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                impact = "Increases" if row['Coefficient'] > 0 else "Decreases"
                report.append(f"{i}. {row['Feature']}: {impact} price (coef: {row['Coefficient']:.4f})")
            report.append("")
        
        report.append("INTERPRETATION")
        report.append("-" * 30)
        
        # Add interpretation based on R²
        r2 = test_metrics.get('r2', 0)
        if r2 >= 0.8:
            report.append("✓ Excellent model performance")
        elif r2 >= 0.6:
            report.append("✓ Good model performance")
        elif r2 >= 0.4:
            report.append("⚠ Moderate model performance")
        else:
            report.append("✗ Poor model performance - consider feature engineering or different algorithm")
        
        # Check overfitting if train_r2 is provided
        if 'train_r2' in test_metrics:
            r2_diff = test_metrics['train_r2'] - r2
            if r2_diff > 0.15:
                report.append(f"⚠ Significant overfitting detected (ΔR² = {r2_diff:.3f})")
            elif r2_diff > 0.1:
                report.append(f"Note: Moderate overfitting (ΔR² = {r2_diff:.3f})")
            else:
                report.append(f"✓ Good generalization (ΔR² = {r2_diff:.3f})")
        
        report.append("")
        report.append("=" * 50)
        
        # Write to file
        try:
            with open(CONFIG.report_file, 'w') as f:
                f.write('\n'.join(report))
            
            print(f"\n✓ Report saved to {CONFIG.report_file}")
            return True
        except Exception as e:
            print(f"✗ Could not write report file: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        print(f"✗ Report generation failed: {e}")
        return False

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Optimized main pipeline with comprehensive error handling"""
    print_section("HOUSE PRICE PREDICTION PIPELINE")
    
    try:
        # 1. Load data
        print("\n1. Loading data...")
        data = load_data()
        
        if data is None or data.empty:
            print("✗ Error: Failed to load data. Exiting.")
            return
        
        print(f"   ✓ Loaded {data.shape[0]} samples, {data.shape[1]} features")
        
        # 2. Clean data
        print("\n2. Cleaning data...")
        try:
            data_clean = clean_data(data)
            print(f"   ✓ Cleaned shape: {data_clean.shape}")
        except Exception as e:
            print(f"✗ Data cleaning failed: {e}")
            return
        
        # 3. Prepare features
        if CONFIG.target_column not in data_clean.columns:
            print(f"\n✗ Error: Target column '{CONFIG.target_column}' not found in data")
            print(f"   Available columns: {list(data_clean.columns)}")
            return
        
        X = data_clean.drop(columns=[CONFIG.target_column])
        y = data_clean[CONFIG.target_column]
        
        if X.empty:
            print("✗ Error: No features available after cleaning")
            return
        
        print(f"   ✓ Target variable: '{CONFIG.target_column}'")
        print(f"   ✓ Features: {X.shape[1]}")
        
        # 4. Feature selection
        print("\n3. Selecting features...")
        try:
            X_selected, selected_features = select_features(X, y)
            print(f"   ✓ Selected {len(selected_features)} features")
        except Exception as e:
            print(f"✗ Feature selection failed: {e}")
            return
        
        # 5. Split data
        print("\n4. Splitting data...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y,
                test_size=CONFIG.test_size,
                random_state=CONFIG.random_state,
                shuffle=True
            )
            print(f"   ✓ Train: {X_train.shape[0]} samples")
            print(f"   ✓ Test:  {X_test.shape[0]} samples")
        except Exception as e:
            print(f"✗ Data splitting failed: {e}")
            return
        
        # 6. Scale features
        print("\n5. Scaling features...")
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print(f"   ✓ Features scaled using StandardScaler")
        except Exception as e:
            print(f"✗ Feature scaling failed: {e}")
            return
        
        # 7. Train model
        print("\n6. Training model...")
        try:
            model = train_model(X_train_scaled, y_train)
            print(f"   ✓ Model: {CONFIG.model_type.upper()}")
        except Exception as e:
            print(f"✗ Model training failed: {e}")
            return
        
        # 8. Evaluate model
        print("\n7. Evaluating model...")
        try:
            y_train_pred, y_test_pred, train_metrics, test_metrics, cv_scores = evaluate_model(
                model, X_train_scaled, X_test_scaled, y_train, y_test, selected_features
            )
            # Include train R² in test_metrics for report
            test_metrics['train_r2'] = train_metrics['r2']
            print(f"   ✓ Evaluation completed")
        except Exception as e:
            print(f"✗ Model evaluation failed: {e}")
            return
        
        # 9. Feature analysis
        print("\n8. Analyzing features...")
        importance = analyze_features(model, selected_features)
        
        # 10. Visualizations
        print("\n9. Generating visualizations...")
        create_visualizations(y_train, y_test, y_train_pred, y_test_pred, importance)
        
        # 11. Save model
        print("\n10. Saving model...")
        save_success = save_model(model, scaler, selected_features, {
            'train': train_metrics,
            'test': test_metrics,
            'cv': {'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores.tolist()}
        })
        
        # 12. Generate report
        print("\n11. Generating report...")
        report_success = generate_report(
            data_clean.shape, selected_features, test_metrics, cv_scores, CONFIG.model_type, importance
        )
        
        # 13. Sample predictions
        print_section("SAMPLE PREDICTIONS")
        try:
            # Use median values for prediction
            sample_median = X_selected.median().values.reshape(1, -1)
            sample_scaled = scaler.transform(sample_median)
            prediction = model.predict(sample_scaled)[0]
            print(f"Predicted price for median house: ${prediction*1000:,.2f}")
            
            # Show a random test sample
            if len(X_test) > 0:
                idx = np.random.randint(0, len(X_test))
                x_sample = X_test.iloc[idx:idx+1].values  # Convert to numpy array
                x_sample_scaled = scaler.transform(x_sample)
                y_pred_sample = model.predict(x_sample_scaled)[0]
                actual_price = y_test.iloc[idx]
                
                error_abs = abs(actual_price - y_pred_sample)
                error_pct = error_abs / actual_price * 100 if actual_price != 0 else float('inf')
                
                print(f"\nRandom test sample #{idx}:")
                print(f"  Actual:    ${actual_price*1000:,.2f}")
                print(f"  Predicted: ${y_pred_sample*1000:,.2f}")
                print(f"  Error:     ${error_abs*1000:,.2f} ({error_pct:.1f}%)")
                
                # Show feature contributions if available
                if importance is not None and not importance.empty and hasattr(model, 'coef_'):
                    print(f"\n  Top contributing features:")
                    coef = np.array(model.coef_).flatten()
                    for feat in importance.head(3)['Feature']:
                        if feat in selected_features:
                            feat_idx = selected_features.index(feat)
                            x_scaled_val = x_sample_scaled[0, feat_idx]
                            contribution = coef[feat_idx] * x_scaled_val
                            direction = "increases" if contribution > 0 else "decreases"
                            print(f"    • {feat}: {direction} price by ~${abs(contribution)*1000:,.0f}")
        except Exception as e:
            print(f"Sample predictions failed: {e}")
        
        print_section("ANALYSIS COMPLETE")
        if save_success:
            print(f"✓ Model saved: {CONFIG.model_file}")
        if report_success:
            print(f"✓ Report saved: {CONFIG.report_file}")
        print(f"✓ Visualization saved: {CONFIG.visualization_file}")
        
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        logger.exception("Pipeline failed with unexpected error")


if __name__ == "__main__":
    main()
