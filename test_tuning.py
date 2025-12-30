import sys
import os
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessor import Preprocessor
from modeling import ModelTrainer

# Setup logging to see progress
logging.basicConfig(level=logging.INFO)

def test_tuning():
    print("Starting Tuning Test...")
    
    # Check data files
    fr_path = 'data/processed/fraud_data_engineered.csv'
    if not os.path.exists(fr_path):
        print(f"Error: Missing data file {fr_path}")
        return

    # Load full dataset to reproduce memory issues
    print(f"Loading data from {fr_path}...")
    df = pd.read_csv(fr_path)
    print(f"Dataset shape: {df.shape}")
    
    # Use a subset for faster verification
    df_subset = df.sample(10000, random_state=42)
    print(f"Using subset shape: {df_subset.shape}")

    preprocessor = Preprocessor()
    trainer = ModelTrainer(random_state=42)

    print("Preparing for modeling...")
    X, y = preprocessor.prepare_for_modeling(df_subset, target_col='class', handle_imbalance=False)
    X_train, X_test, y_train, y_test = preprocessor.stratified_split(X, y, test_size=0.2)
    
    print(f"Train set shape: {X_train.shape}")
    
    print("\nAttempting Hyperparameter Tuning (tune=True)...")
    try:
        # Override n_iter for hyperparameter tuning test
        # We'll use a local version of RandomizedSearchCV logic for the test
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from joblib import parallel_backend
        
        param_dist = {
            'n_estimators': [10, 20],
            'max_depth': [None, 5],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=2, 
            cv=cv, scoring='f1', n_jobs=1, random_state=42
        )
        
        with parallel_backend('sequential'):
            search.fit(X_train, y_train)
            
        print(f"Tuning successful! Best params: {search.best_params_}")
    except Exception as e:
        print(f"\nTuning FAILED with error: {type(e).__name__}")
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tuning()
