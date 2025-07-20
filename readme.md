# 🕵️ Fraud Detection using Machine Learning

This project presents an end-to-end fraud detection pipeline built using machine learning techniques. The goal is to detect fraudulent transactions using structured data and interpretable models. The process includes data cleaning, feature engineering, modeling, and SHAP-based explainability.

---

## 📁 Project Structure

```
WEEK-8-9/
├── .github/
│   └── workflow/
│       └── ci.yml
├── Data/
│   ├── processed/
│   │   ├── creditcard_scaler.joblib
│   │   ├── creditcard_X_test.csv
│   │   ├── creditcard_X_train.csv
│   │   ├── creditcard_y_test.csv
│   │   ├── creditcard_y_train.csv
│   │   ├── fraud_scaler.joblib
│   │   ├── fraud_X_test.csv
│   │   ├── fraud_X_train.csv
│   │   ├── fraud_y_test.csv
│   │   └── fraud_y_train.csv
│   └── raw/
│       ├── creditcard.csv
│       ├── Fraud_Data.csv
│       └── IpAddress_to_Country.csv
├── notebooks/
│   ├── data.ipynb
│   └── model.ipynb
└── .gitignore
```

---

## 📌 Project Highlights

### 🧹 Data Analysis and Preprocessing

* **Handled missing values** by imputation or removal, depending on context.
* **Removed duplicates** and corrected data types for consistency.
* **Exploratory Data Analysis (EDA)** included both univariate and bivariate visualizations to understand distributions and relationships.
* **Merged geolocation data** by converting IP addresses to integers and joining `Fraud_Data.csv` with `IpAddress_to_Country.csv`.
* **Feature engineering** included:

  * `hour_of_day`, `day_of_week`, and `time_since_signup` (from timestamp differences)
  * Transaction frequency and velocity features
* **Class imbalance addressed** using sampling techniques like SMOTE or undersampling — applied only to training data.
* **Normalization and encoding**:

  * Scaling performed using `StandardScaler` or `MinMaxScaler`
  * One-hot encoding applied to categorical features

### 🤖 Model Building and Evaluation

* **Train-test split** done after separating features and targets (`Class` or `class`)
* Built two models:

  * **Logistic Regression**: for interpretability and as a baseline
  * **Ensemble Model**: Random Forest, LightGBM, or XGBoost for improved performance
* **Evaluated using**:

  * Confusion Matrix
  * F1-Score
  * Precision-Recall AUC (more appropriate for imbalanced datasets)
* Final model chosen based on performance and interpretability

### 📉 Model Explainability with SHAP

* **SHAP Summary Plots** used to identify globally important features
* **SHAP Force Plots** generated for local, instance-level explanations
* Key fraud indicators identified included:

  * Suspicious transaction timing
  * Short time between signup and purchase
  * High-frequency user activity
  * Anomalous geolocations

---

## 🛠️ Tech Stack

* **Languages**: Python
* **Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, shap, seaborn, matplotlib, joblib
* **Models**: Logistic Regression, Random Forest / XGBoost / LightGBM
* **Environment**: Jupyter Notebook

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/Crackingastro/Week-8-9
   cd fraud-detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**

   ```bash
   jupyter notebook
   ```

---

## ✅ Deliverables

* Cleaned datasets in `Data/processed/`
* Trained models and scalers (`.joblib`)
* Visual SHAP explanations
* Model evaluation summaries and fraud insights

---

**Made with ❤️ by Cracking Astro**
