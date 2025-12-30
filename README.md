# Fraud Detection Model

## Project Overview
This project aims to detect fraudulent transactions using machine learning techniques. The analysis combines multiple datasets including transaction data, IP address mappings, and credit card fraud records to build a comprehensive fraud detection system.

## Business Objective
Adey Innovations Inc. aims to protect its e-commerce and banking ecosystem by detecting fraudulent transactions in real-time. This project focuses on:
- **Financial Security**: Identifying and blocking fraudulent activities to prevent monetary loss.
- **User Trust**: Balancing rigorous security with a frictionless consumer experience by minimizing false positives.
- **Pattern Recognition**: Analyzing geographic, temporal, and user behavioral data to stay ahead of sophisticated fraud patterns.

📊 **View the [Interim Report](file:///C:/Users/Mohammed/.gemini/antigravity/brain/1249fc4d-f56d-446a-b4f5-bd5d3c162eb8/interim_report.md)** for a detailed analysis of our progress.

## Datasets
The project uses three main datasets:

1. **Fraud_Data.csv**: E-commerce transaction data
   - 151,112 transactions
   - Features: user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class

2. **IpAddress_to_Country.csv**: IP geolocation mapping
   - 138,846 IP ranges mapped to countries
   - Features: lower_bound_ip_address, upper_bound_ip_address, country

3. **creditcard.csv**: Credit card transaction data
   - 284,807 transactions
   - Features: Time, V1-V28 (PCA-transformed), Amount, Class

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection-model

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
fraud-detection-model/
├── data/
│   ├── raw/              # Raw data files (git-ignored)
│   └── processed/        # Cleaned and processed data
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_class_imbalance_handling.ipynb
│   └── 05_model_training_and_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineer.py
│   ├── preprocessor.py
│   └── modeling.py
├── tests/                # Unit tests
├── models/               # Trained models
├── docs/                 # Documentation
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

### Running Notebooks
Execute notebooks in order:
```bash
jupyter notebook
# Open and run: 01_data_cleaning.ipynb → 02_exploratory_data_analysis.ipynb → ...
```

### Using Python Scripts
```python
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner

# Load data
loader = DataLoader()
df = loader.load_fraud_data('data/raw/Fraud_Data.csv')

# Clean data
cleaner = DataCleaner()
df_clean = cleaner.handle_missing_values(df)
```

## Key Findings (Interim Results)
- Class imbalance: Approximately 90% non-fraud, 10% fraud transactions
- Time-based patterns: Higher fraud rates during specific hours
- Geographic insights: Certain regions show elevated fraud risk
- Feature importance: Time since signup and transaction amount are strong indicators
- **Task 2a Results**: Successfully trained a baseline Logistic Regression model with reporting on AUC-PR, F1-Score, and Confusion Matrix.

## Task 2: Model Training and Evaluation (Interim-2)
Our latest progress includes both Task 2a and Task 2b requirements:
- **Task 2a (Baseline)**: Established baseline performance using Logistic Regression with stratified splitting.
- **Task 2b (Ensemble)**: Developed a tuned Random Forest model with 5-fold Stratified Cross-Validation.
- **Evaluation**: Comprehensive comparison of models using AUC-PR, F1-Score, and reporting mean/std across CV folds.

## Task 3: Model Explainability (Completed)
Our model interpretation phase (Task 3) is complete, featuring:
- **Global Interpretability**: SHAP Summary plots showing feature impact across the entire dataset.
- **Local Interpretability**: Detailed explanations for True Positive, False Positive, and False Negative predictions.
- **Business Insights**: Developed 4 key actionable recommendations for fraud prevention based on model drivers like `time_since_signup`.

All Task 3 artifacts are available on the `task-3` branch.

## Next Steps
1. API development for real-time fraud prediction
2. Containerization for deployment
4. Continuous integration and monitoring

## Contributors
- Mohammed Sultan

## License
This project is part of an academic/professional assessment.
