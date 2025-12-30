import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.explainability import ModelExplainer

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'time_since_signup': np.random.rand(100),
        'purchase_value': np.random.rand(100),
        'age': np.random.randint(18, 80, 100),
        'source': np.random.randint(0, 3, 100)
    })
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

def test_model_explainer_init(trained_model, sample_data):
    X, _ = sample_data
    explainer = ModelExplainer(trained_model, X)
    assert explainer.model == trained_model
    assert explainer.feature_names == list(X.columns)
    assert explainer.explainer is not None

def test_compute_shap_values(trained_model, sample_data):
    X, _ = sample_data
    explainer = ModelExplainer(trained_model, X)
    shap_values = explainer.compute_shap_values(X.head(10))
    
    # SHAP values for RF (binary) can be a list of two arrays or a single array 
    # depending on shap version, but usually it's list [negative, positive]
    if isinstance(shap_values, list):
        assert len(shap_values) == 2
        assert shap_values[1].shape == (10, 4)
    else:
        assert shap_values.shape == (10, 4)

def test_get_top_features(trained_model, sample_data):
    X, _ = sample_data
    explainer = ModelExplainer(trained_model, X)
    top_features = explainer.get_top_features(X.head(10), top_n=2)
    
    assert len(top_features) == 2
    assert 'feature' in top_features.columns
    assert 'mean_abs_shap' in top_features.columns

def test_generate_business_recommendations(trained_model, sample_data):
    X, _ = sample_data
    explainer = ModelExplainer(trained_model, X)
    top_features = pd.DataFrame({
        'feature': ['time_since_signup', 'purchase_value'],
        'mean_abs_shap': [0.5, 0.4]
    })
    recs = explainer.generate_business_recommendations(top_features)
    
    assert len(recs) >= 2
    assert any("Monitor New Accounts Closely" in r for r in recs)
    assert any("Transaction Amount Monitoring" in r for r in recs)

def test_find_prediction_examples(trained_model, sample_data):
    X, y = sample_data
    explainer = ModelExplainer(trained_model, X)
    y_pred = trained_model.predict(X)
    
    examples = explainer.find_prediction_examples(X, y, y_pred)
    assert isinstance(examples, dict)
    # Check that at least one of the keys is present (depends on random data)
    assert any(k in examples for k in ['true_positive', 'false_positive', 'false_negative', 'true_negative'])
