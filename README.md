# 🔍 Fraud Detection with Machine Learning & Streamlit

> An end-to-end binary classification project that screens financial transactions and flags those that may be fraudulent.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

## ✨ Project overview

Financial fraud is rare but costly. This project uses transaction details—such as transaction type, amount, and the sender's and receiver's balances—to predict whether a transaction should be flagged as potentially fraudulent.

The project follows a complete machine-learning workflow:

```text
Transaction data → exploration & preprocessing → ML pipeline → saved model → Streamlit prediction app
```

Users enter transaction information in a simple web form and receive an instant prediction:

- `1` — potentially fraudulent transaction
- `0` — transaction not predicted as fraudulent

## 🧠 How it works

The trained model is a **Logistic Regression** classifier packaged in a scikit-learn `Pipeline`. Keeping preprocessing and the model together helps ensure that the Streamlit app transforms new inputs in exactly the same way as the training data.

| Stage | Implementation |
|---|---|
| Data split | 70% training / 30% testing, using stratified sampling |
| Numeric features | `StandardScaler` |
| Categorical feature | One-hot encoding for transaction type |
| Class imbalance | `class_weight="balanced"` in Logistic Regression |
| Deployment | Streamlit app loads `fraud_detection_pipeline.pkl` with Joblib |

### Model inputs

| Feature | Description |
|---|---|
| `type` | Categorical transaction type |
| `amount` | Monetary value of the transaction |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before and after the transaction |
| `oldbalanceDest` / `newbalanceDest` | Receiver balance before and after the transaction |

Account identifiers and the existing-system fraud flag are excluded from model training to avoid using identifiers as direct predictive inputs. The notebook also explores balance-difference features as part of the analysis.

## 📊 Dataset

The project uses a financial transaction fraud dataset published on Kaggle:

- **Records:** 6,362,620 transactions
- **Target:** `isFraud` (`1` = fraud, `0` = non-fraud)
- **Class distribution:** 8,213 fraud cases, approximately **0.13%** of all transactions
- **Problem type:** Binary classification

Dataset: [Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)

## 📈 Evaluation results

On the notebook's held-out test set, the model produced:

| Metric | Non-fraud (0) | Fraud (1) |
|---|---:|---:|
| Precision | 1.00 | 0.02 |
| Recall | 0.95 | 0.95 |
| F1-score | 0.97 | 0.04 |

Overall accuracy was **95%**. However, because fraud is so rare, accuracy alone is not a reliable measure of fraud-detection quality. The model achieves high fraud recall—meaning it identifies many actual fraud cases—but its low fraud precision means it also generates many false alerts. This is a useful baseline model, not a production-ready fraud decision system.

## 🖥️ Run locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Fraud-Detection-Using-ML-and-Streamlit.git
cd Fraud-Detection-Using-ML-and-Streamlit
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

### 4. Start the app

```bash
streamlit run fraud_detection.py
```

Open the local address shown in the terminal (normally `http://localhost:8501`) and enter a transaction to make a prediction.

## 📁 Project structure

```text
├── fraud_analysis.ipynb                 # Data exploration, preprocessing, training, and evaluation
├── fraud_detection.py                   # Streamlit prediction interface
├── fraud_detection_pipeline.pkl         # Saved preprocessing + Logistic Regression pipeline
├── fraud-detection-dataset-metadata.json # Dataset metadata and source information
└── README.md
```

## 🎯 Key learning outcomes

- Performed exploratory data analysis on a large financial-transaction dataset.
- Built a reproducible scikit-learn preprocessing and classification pipeline.
- Handled a highly imbalanced classification problem using class weighting.
- Evaluated the model using precision, recall, F1-score, confusion matrix, and accuracy.
- Deployed the trained model as an interactive Streamlit application.

## 🔮 Suggested next steps

- Compare the baseline with tree-based models such as Random Forest, XGBoost, or LightGBM.
- Tune the classification threshold to balance fraud recall against false-positive alerts.
- Use precision-recall curves, PR-AUC, and cost-based metrics rather than accuracy alone.
- Add the engineered balance-difference features to the deployed pipeline and compare results.
- Add unit tests, dependency pinning, model versioning, and input validation before production use.

## ⚠️ Disclaimer

This repository is an educational machine-learning project. Its output should be treated as a screening signal, not an automatic financial decision. Real fraud systems require monitoring, security controls, fairness checks, human review, and ongoing model validation.

---

Built with Python, scikit-learn, and Streamlit.
