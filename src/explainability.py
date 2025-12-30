"""
Explainability Module for Fraud Detection Project.

This module provides SHAP-based model interpretability and business insights.
Implements global and local feature importance analysis.
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Generate model explanations using SHAP.
    
    Provides methods for:
    - Computing SHAP values
    - Generating summary and force plots
    - Extracting top features
    - Creating business recommendations
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained tree-based model (e.g., RandomForest)
            X_train: Training data used for background distribution
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = list(X_train.columns)
        
        logger.info("Initializing SHAP TreeExplainer...")
        try:
            self.explainer = shap.TreeExplainer(model)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Features to explain
            
        Returns:
            Array of SHAP values
        """
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X)
        logger.info("SHAP values computed successfully")
        return shap_values
    
    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None, 
                     max_display: int = 20) -> None:
        """
        Generate SHAP summary plot (global feature importance).
        
        Args:
            X: Features to explain
            save_path: Optional path to save the plot
            max_display: Maximum number of features to display
        """
        logger.info("Generating SHAP summary plot...")
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        # For binary classification, use positive class (index 1)
        if isinstance(shap_values, list):
            # For binary classification, use positive class (index 1) if available
            val = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap.summary_plot(val, X, max_display=max_display, show=False)
        elif len(shap_values.shape) == 3:
            # Handle 3D array (n_samples, n_features, n_classes)
            shap.summary_plot(shap_values[:, :, 1], X, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        
        plt.title("SHAP Summary Plot - Feature Impact on Fraud Prediction", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_bar_summary(self, X: pd.DataFrame, save_path: Optional[str] = None,
                         max_display: int = 15) -> None:
        """
        Generate SHAP bar summary plot (mean absolute SHAP values).
        
        Args:
            X: Features to explain
            save_path: Optional path to save the plot
            max_display: Maximum number of features to display
        """
        logger.info("Generating SHAP bar summary plot...")
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            val = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap.summary_plot(val, X, plot_type="bar", 
                            max_display=max_display, show=False)
        elif len(shap_values.shape) == 3:
            shap.summary_plot(shap_values[:, :, 1], X, plot_type="bar",
                            max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X, plot_type="bar", 
                            max_display=max_display, show=False)
        
        plt.title("Mean Absolute SHAP Values - Feature Importance", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Bar summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_force(self, X: pd.DataFrame, idx: int, 
                   save_path: Optional[str] = None) -> None:
        """
        Generate SHAP force plot for a single prediction.
        
        Args:
            X: Features DataFrame
            idx: Index of sample to explain
            save_path: Optional path to save the plot
        """
        logger.info(f"Generating force plot for sample {idx}...")
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(20, 3))
        if isinstance(shap_values, list):
            expected_val = self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, "__len__") and len(self.explainer.expected_value) > 1 else self.explainer.expected_value
            val = shap_values[1][idx] if len(shap_values) > 1 else shap_values[0][idx]
            shap.force_plot(
                expected_val,
                val,
                X.iloc[idx],
                matplotlib=True,
                show=False
            )
        elif len(shap_values.shape) == 3:
            # Handle 3D array
            shap.force_plot(
                self.explainer.expected_value[1],
                shap_values[idx, :, 1],
                X.iloc[idx],
                matplotlib=True,
                show=False
            )
        else:
            shap.force_plot(
                self.explainer.expected_value,
                shap_values[idx],
                X.iloc[idx],
                matplotlib=True,
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Force plot saved to {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, X: pd.DataFrame, idx: int,
                       save_path: Optional[str] = None) -> None:
        """
        Generate SHAP waterfall plot for a single prediction.
        
        Args:
            X: Features DataFrame  
            idx: Index of sample to explain
            save_path: Optional path to save the plot
        """
        logger.info(f"Generating waterfall plot for sample {idx}...")
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            expected_val = self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, "__len__") and len(self.explainer.expected_value) > 1 else self.explainer.expected_value
            val = shap_values[1][idx] if len(shap_values) > 1 else shap_values[0][idx]
            shap.waterfall_plot(
                shap.Explanation(
                    values=val,
                    base_values=expected_val,
                    data=X.iloc[idx].values,
                    feature_names=self.feature_names
                ),
                show=False
            )
        elif len(shap_values.shape) == 3:
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[idx, :, 1],
                    base_values=self.explainer.expected_value[1],
                    data=X.iloc[idx].values,
                    feature_names=self.feature_names
                ),
                show=False
            )
        else:
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[idx],
                    base_values=self.explainer.expected_value,
                    data=X.iloc[idx].values,
                    feature_names=self.feature_names
                ),
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def get_top_features(self, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N features by mean absolute SHAP value.
        
        Args:
            X: Features to analyze
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        logger.info(f"Computing top {top_n} features...")
        shap_values = self.compute_shap_values(X)
        
        if isinstance(shap_values, list):
            importance = np.abs(shap_values[1]).mean(axis=0) if len(shap_values) > 1 else np.abs(shap_values[0]).mean(axis=0)
        elif len(shap_values.shape) == 3:
            importance = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        top_features = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': importance
        }).sort_values('mean_abs_shap', ascending=False).head(top_n)
        
        top_features['rank'] = range(1, len(top_features) + 1)
        top_features = top_features[['rank', 'feature', 'mean_abs_shap']]
        
        logger.info(f"Top feature: {top_features.iloc[0]['feature']}")
        return top_features
    
    def compare_with_builtin_importance(self, X: pd.DataFrame, 
                                        top_n: int = 10) -> pd.DataFrame:
        """
        Compare SHAP importance with model's built-in feature importance.
        
        Args:
            X: Features to analyze
            top_n: Number of top features to compare
            
        Returns:
            DataFrame comparing SHAP and built-in importance
        """
        logger.info("Comparing SHAP with built-in feature importance...")
        
        # Get SHAP-based importance
        shap_top = self.get_top_features(X, top_n=top_n)
        
        # Get built-in importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            builtin_importance = pd.DataFrame({
                'feature': X.columns,
                'builtin_importance': self.model.feature_importances_
            }).sort_values('builtin_importance', ascending=False)
            
            # Add rank for built-in importance
            builtin_importance['builtin_rank'] = range(1, len(builtin_importance) + 1)
            
            # Merge the two
            comparison = shap_top.merge(
                builtin_importance[['feature', 'builtin_importance', 'builtin_rank']],
                on='feature',
                how='left'
            )
            comparison.columns = ['shap_rank', 'feature', 'shap_importance', 
                                 'builtin_importance', 'builtin_rank']
            
            return comparison
        else:
            logger.warning("Model does not have built-in feature_importances_")
            return shap_top
    
    def find_prediction_examples(self, X: pd.DataFrame, y_true: pd.Series,
                                  y_pred: np.ndarray) -> Dict[str, int]:
        """
        Find indices of True Positive, False Positive, and False Negative cases.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with indices for TP, FP, FN examples
        """
        y_true_arr = y_true.values if hasattr(y_true, 'values') else y_true
        
        tp_mask = (y_true_arr == 1) & (y_pred == 1)
        fp_mask = (y_true_arr == 0) & (y_pred == 1)
        fn_mask = (y_true_arr == 1) & (y_pred == 0)
        
        examples = {}
        
        if tp_mask.any():
            examples['true_positive'] = np.where(tp_mask)[0][0]
            logger.info(f"True Positive example found at index {examples['true_positive']}")
        
        if fp_mask.any():
            examples['false_positive'] = np.where(fp_mask)[0][0]
            logger.info(f"False Positive example found at index {examples['false_positive']}")
        
        if fn_mask.any():
            examples['false_negative'] = np.where(fn_mask)[0][0]
            logger.info(f"False Negative example found at index {examples['false_negative']}")
        
        return examples
    
    def generate_business_recommendations(self, top_features: pd.DataFrame) -> List[str]:
        """
        Generate actionable business recommendations from top features.
        
        Args:
            top_features: DataFrame with top features from get_top_features()
            
        Returns:
            List of business recommendation strings
        """
        logger.info("Generating business recommendations...")
        recommendations = []
        seen_categories = set()
        
        feature_to_recommendation = {
            'time_since_signup': (
                "time_monitoring",
                "🔍 **Monitor New Accounts Closely**: Transactions from recently created "
                "accounts are strong fraud indicators. Implement enhanced verification for "
                "accounts less than 24-48 hours old, such as email confirmation, phone "
                "verification, or lower transaction limits."
            ),
            'hour': (
                "time_patterns",
                "⏰ **Implement Time-Based Risk Scoring**: Certain hours show elevated "
                "fraud rates (typically late night/early morning). Apply enhanced scrutiny "
                "to transactions during high-risk periods and consider adaptive authentication."
            ),
            'purchase_hour': (
                "time_patterns",
                "⏰ **Implement Time-Based Risk Scoring**: Certain hours show elevated "
                "fraud rates (typically late night/early morning). Apply enhanced scrutiny "
                "to transactions during high-risk periods and consider adaptive authentication."
            ),
            'country': (
                "geo_monitoring", 
                "🌍 **Geolocation-Based Alerts**: Flag transactions from high-risk regions "
                "or when user location differs significantly from historical patterns. "
                "Consider implementing velocity checks for cross-border transactions."
            ),
            'amount': (
                "transaction_monitoring",
                "💰 **Transaction Amount Monitoring**: Large or unusual transaction amounts "
                "warrant additional review. Set dynamic thresholds based on user history "
                "and implement stepped limits for new users."
            ),
            'purchase_value': (
                "transaction_monitoring",
                "💰 **Transaction Amount Monitoring**: Large or unusual transaction amounts "
                "warrant additional review. Set dynamic thresholds based on user history "
                "and implement stepped limits for new users."
            ),
            'device': (
                "device_fingerprinting",
                "📱 **Device Fingerprinting**: Multiple accounts using the same device ID "
                "is a red flag. Implement device reputation scoring and flag suspicious "
                "device-to-account ratios."
            ),
            'user_transaction_count': (
                "velocity_checks",
                "📊 **Transaction Velocity Monitoring**: Unusual spikes in transaction "
                "frequency indicate potential fraud. Implement real-time velocity checks "
                "with dynamic thresholds based on user behavior history."
            ),
            'device_transaction_count': (
                "velocity_checks",
                "📊 **Transaction Velocity Monitoring**: Unusual spikes in transaction "
                "frequency indicate potential fraud. Implement real-time velocity checks "
                "with dynamic thresholds based on user behavior history."
            ),
            'browser': (
                "browser_analysis",
                "🌐 **Browser/User-Agent Analysis**: Fraudsters often use specific browsers "
                "or automation tools. Monitor for suspicious user-agent strings and "
                "implement browser fingerprinting."
            ),
            'source': (
                "traffic_monitoring",
                "📈 **Traffic Source Monitoring**: Certain acquisition channels may have "
                "higher fraud rates. Track fraud by source and implement source-specific "
                "risk scoring."
            ),
            'day_of_week': (
                "pattern_analysis",
                "📅 **Behavioral Pattern Analysis**: Fraud often follows specific patterns "
                "(e.g., weekend activity for new accounts). Build user behavior profiles "
                "and flag deviations from established patterns."
            ),
            'age': (
                "demographic_analysis",
                "👤 **Demographic Risk Profiling**: Certain age groups may show different "
                "fraud patterns. Use demographic data as one signal in a multi-factor "
                "risk assessment, while ensuring compliance with regulations."
            ),
        }
        
        for _, row in top_features.iterrows():
            feature = row['feature'].lower()
            
            for key, (category, recommendation) in feature_to_recommendation.items():
                if key in feature and category not in seen_categories:
                    recommendations.append(recommendation)
                    seen_categories.add(category)
                    break
        
        # Add general recommendations if we have few specific ones
        if len(recommendations) < 3:
            recommendations.append(
                "🔒 **Multi-Factor Authentication**: Implement adaptive MFA based on "
                "transaction risk score. High-risk transactions should require additional "
                "verification steps."
            )
            recommendations.append(
                "📊 **Real-Time Scoring**: Deploy the fraud model for real-time transaction "
                "scoring. Set score thresholds for auto-approve, review, and block actions."
            )
        
        logger.info(f"Generated {len(recommendations)} business recommendations")
        return recommendations[:6]  # Return top 6 recommendations


def create_importance_comparison_plot(shap_importance: pd.DataFrame,
                                       builtin_importance: np.ndarray,
                                       feature_names: List[str],
                                       save_path: Optional[str] = None) -> None:
    """
    Create side-by-side comparison of SHAP vs built-in feature importance.
    
    Args:
        shap_importance: DataFrame from get_top_features()
        builtin_importance: Array of built-in feature importances
        feature_names: List of feature names
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # SHAP importance (left)
    top_n = min(15, len(shap_importance))
    shap_top = shap_importance.head(top_n)
    
    axes[0].barh(range(top_n), shap_top['mean_abs_shap'].values[::-1], color='steelblue')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(shap_top['feature'].values[::-1])
    axes[0].set_xlabel('Mean |SHAP Value|')
    axes[0].set_title('SHAP Feature Importance', fontsize=14)
    
    # Built-in importance (right)
    builtin_df = pd.DataFrame({
        'feature': feature_names,
        'importance': builtin_importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    axes[1].barh(range(top_n), builtin_df['importance'].values[::-1], color='coral')
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(builtin_df['feature'].values[::-1])
    axes[1].set_xlabel('Feature Importance (Gini)')
    axes[1].set_title('Built-in Feature Importance (Random Forest)', fontsize=14)
    
    plt.suptitle('SHAP vs Built-in Feature Importance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()
