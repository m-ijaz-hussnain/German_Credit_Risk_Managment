# 🏦 German Credit Risk Analysis

An end-to-end machine learning project that analyzes and predicts credit risk using the German Credit Dataset. The objective is to help financial institutions assess the likelihood of a customer defaulting on a loan by building robust classification models and visualizing insights via Power BI dashboards.

---

## 📌 Project Overview

- **Objective:** Predict whether a loan applicant is likely to default.
- **Dataset:** German Credit Data — includes features like loan history, credit amount, job type, personal status, and more.
- **Goal:** Develop a reliable credit risk classifier and present model outcomes in a business-friendly format.

---

## 🧪 Key Features

### 🔍 Exploratory Data Analysis (EDA)
- Performed detailed EDA using `Pandas`, `Matplotlib`, and `Seaborn`.
- Uncovered patterns in customer demographics and financial behavior.
- Handled missing values and outliers.

### 🤖 Machine Learning
- Trained multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Achieved **>85% accuracy** with well-balanced precision and recall.
- Feature selection and engineering to improve model performance.

### ✅ Model Evaluation
- Metrics used:
  - Confusion Matrix
  - ROC-AUC Curve
  - Precision, Recall, F1 Score
- Applied **cross-validation** and **GridSearchCV** for hyperparameter tuning.

### 📊 Dashboarding with Power BI
- Built an interactive Power BI dashboard showing:
  - Risk predictions
  - Feature importance
  - Customer segmentation
- Designed for use by non-technical stakeholders.

---

## 🧰 Tech Stack

- **Languages & Libraries:** Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Visualization:** Power BI
- **Tools:** Jupyter Notebook, Power BI Desktop

---
german-credit-risk-analysis/
├── data/
│ └── german_credit_data.csv
├── notebooks/
│ ├── 01_eda.ipynb
│ └── 02_modeling.ipynb
├── visuals/
│ └── credit_risk_dashboard.pbix
├── src/
│ └── utils.py
├── README.md
└── requirements.txt



---

## 📈 Results

- Best model achieved over **85% accuracy**.
- High interpretability with feature importance visualizations.
- Dashboard provides **real-time filtering and drill-down** into customer segments.

---

## 🚀 Future Enhancements

- Deploy model as a REST API using Flask/FastAPI.
- Integrate SHAP for explainability.
- Set up scheduled retraining and monitoring pipeline.

---

## 📎 Resources

- 📁 [Original Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- 📊 Power BI Dashboard (optional: upload & share link if public)
- 📘 Documentation (optional)

---

## 🙌 Acknowledgements

- UCI Machine Learning Repository for the dataset.
- Open-source libraries that made this project possible.

---

## 📬 Contact

For questions or collaborations:  
**M IJAZ HUSSNAIN**  
mijazhussnain83@gmail.com  

---

⭐ If you found this project helpful, consider giving it a star!

## 📂 Project Structure

