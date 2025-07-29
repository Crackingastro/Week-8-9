import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
from IPython.display import display

# 1. Load and preprocess data
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    # Convert object columns to numeric where possible
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # If conversion fails, drop the column or use encoding
            df = df.drop(col, axis=1)
    return df

fraud_X_train = load_and_preprocess_data('../Data/processed/fraud_X_train.csv')
fraud_X_test = load_and_preprocess_data('../Data/processed/fraud_X_test.csv')
fraud_y_train = pd.read_csv('../Data/processed/fraud_y_train.csv').squeeze()
fraud_y_test = pd.read_csv('../Data/processed/fraud_y_test.csv').squeeze()

cc_X_train = load_and_preprocess_data('../Data/processed/creditcard_X_train.csv')
cc_X_test = load_and_preprocess_data('../Data/processed/creditcard_X_test.csv')
cc_y_train = pd.read_csv('../Data/processed/creditcard_y_train.csv').squeeze()
cc_y_test = pd.read_csv('../Data/processed/creditcard_y_test.csv').squeeze()

def shap_analysis(model, X_train, y_train, X_test, model_name, dataset_name):
    """
    Perform SHAP analysis on a trained model and generate interpretation plots
    
    Parameters:
    - model: Trained model (must support SHAP)
    - X_train: Training features (numeric only)
    - y_train: Training labels
    - X_test: Test features to explain (numeric only)
    - model_name: Name of model for titles
    - dataset_name: Name of dataset for titles
    """
    
    # Ensure data is numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"\nSHAP Analysis for {model_name} - {dataset_name}")
    print("="*60)
    
    # Train the model
    model.fit(X_train, y_train)
    
    try:
        # Initialize SHAP explainer
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")
        print("Trying TreeExplainer instead...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
    # 1. Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"{model_name} - {dataset_name}\nGlobal Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 2. Detailed Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"{model_name} - {dataset_name}\nFeature Impact Direction", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 3. Force Plot
    print("\nForce Plot for First Test Instance:")
    shap.plots.force(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
    
    # 4. Dependence Plot
    if hasattr(shap_values, 'values'):
        mean_shap = np.abs(shap_values.values).mean(0)
    else:
        mean_shap = np.abs(shap_values).mean(0)
    top_feature = X_test.columns[np.argmax(mean_shap)]
    print(f"\nDependence Plot for Top Feature ('{top_feature}'):")
    shap.dependence_plot(top_feature, shap_values, X_test, show=False)
    plt.title(f"{model_name} - {dataset_name}\nSHAP Dependence Plot", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return shap_values

# Model parameters
best_params_fraud = {
    "n_estimators": 252,
    "max_depth": 8,
    "learning_rate": 0.19400372923812143,
    "subsample": 0.8978799950358755,
    "colsample_bytree": 0.7057902648625826,
    "gamma": 0.019502545482357816,
    "min_child_weight": 1}

best_params_cc = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.16405246147546276,
    "subsample": 0.7354474953142038,
    "colsample_bytree": 0.7679787915015344,
    "gamma": 1.400025824187118,
    "min_child_weight": 2}

# Initialize models
print("\nInitializing models...")
xgb_fraud = XGBClassifier(**best_params_fraud, random_state=42, eval_metric='logloss')
xgb_cc = XGBClassifier(**best_params_cc, random_state=42, eval_metric='logloss')

# Run SHAP analysis
print("\nRunning SHAP analysis for Fraud Data...")
shap_fraud = shap_analysis(xgb_fraud, fraud_X_train, fraud_y_train, fraud_X_test,
                          "XGBoost", "Fraud Data")

print("\nRunning SHAP analysis for Credit Card Data...")
shap_cc = shap_analysis(xgb_cc, cc_X_train, cc_y_train, cc_X_test,
                       "XGBoost", "Credit Card Data")