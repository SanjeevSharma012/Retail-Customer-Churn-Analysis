# Retail Customer Churn Analysis

## Project Overview

This project focuses on building a machine learning model to predict customer churn for a retail business, identifying key factors influencing churn, and translating these insights into actionable retention strategies. Leveraging a real-world transactional dataset (Online Retail II from Kaggle), this project demonstrates a comprehensive data science workflow from data acquisition and cleaning to feature engineering, advanced machine learning modeling (Gradient Boosting), and interactive visualization using Streamlit.

The primary objective is to empower business stakeholders with a data-driven tool to proactively identify customers at high risk of churning and understand why they are churning, enabling targeted interventions to improve customer retention and lifetime value.

---

## Problem Statement

Customer churn is a significant challenge for retail businesses, directly impacting revenue and growth. Identifying at-risk customers early allows businesses to implement targeted retention campaigns (e.g., personalized offers, improved customer service) before customers are lost. This project addresses this by:

- Developing a predictive model for customer churn.
- Uncovering the most influential behavioral and transactional factors that contribute to churn.
- Providing an interactive platform for stakeholders to explore churn insights and identify at-risk customers.

---

## Dataset

- **Source**: [Online Retail II - Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- **Description**: A transactional dataset containing all purchases made by customers of a UK-based online retail store. It spans a period from December 2009 to December 2011.
- **Key Columns**: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `CustomerID`, `Country`.

> **Note on Churn Definition**:  
> As the dataset does not contain an explicit churn label, churn was defined based on customer inactivity. A customer was considered churned if they had no transactions within a 3-month churn observation window following their last activity in the analysis period.

---

## Project Phases & Methodology

### 1. Project Planning & Setup
- Defined project scope, objectives, initial tools, and established the repository structure.

### 2. Data Acquisition & Understanding
- Acquired the Online Retail II dataset.
- Performed initial data loading, cleaning (handling missing `CustomerID`s, invalid quantities/prices, cancelled orders).
- Conducted comprehensive Exploratory Data Analysis (EDA) to understand data distributions and temporal trends.

### 3. Data Preprocessing & Feature Engineering
- Defined a 3-month churn window and an observation period.
- Engineered crucial customer-level features:
  - **RFM (Recency, Frequency, Monetary)**: Calculated based on customer activity within the observation period.
  - **Tenure**: Days since the customer's first purchase.
- Prepared the `Country` feature (grouping top countries, one-hot encoding).
- Created the binary `is_churned` target variable.

### 4. Modeling & Evaluation
- Split the feature-engineered data into training and testing sets, stratifying by churn status.
- Trained and evaluated multiple classification models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Selected Gradient Boosting Classifier for its superior performance (highest ROC-AUC).
- Performed hyperparameter tuning (GridSearchCV) to optimize ROC-AUC.
- Evaluated the final model using metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix
- Identified Feature Importances to understand key churn drivers.

### 5. Visualization & Communication
- Developed an interactive **Streamlit** dashboard to visualize churn insights:
  - Displays overall churn rates, churn by country.
  - Highlights top churn drivers.
  - Allows identifying at-risk customers based on an adjustable probability threshold.

---

## Technologies Used

- **Programming Language**: Python 3.x
- **Data Manipulation & Analysis**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GridSearchCV`, `train_test_split`, `metrics`)
- **Model Persistence**: `joblib`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `streamlit`
- **Version Control**: Git, GitHub
- **Development Environment**: Jupyter Notebook

---

## Results & Insights (Example - UPDATE WITH YOUR ACTUAL VALUES)

The Tuned Gradient Boosting Classifier was chosen as the final model due to its robust performance, achieving excellent discriminative power for churn prediction.

**Key Model Performance on Test Set**:

- **Accuracy**: 0.7419
- **Precision**: 0.7815
- **Recall**: 0.7553
- **F1-Score**: 0.7682
- **ROC-AUC**: 0.8107
- **Confusion Matrix**: `[[415 158], [183 565]]`

**Top Churn Drivers (from Gradient Boosting Feature Importance):**
- **Recency** – Customers who haven't purchased recently are at higher risk.
- **Monetary** – Lower spending customers might be more prone to churn.
- **Frequency** – Less frequent buyers have higher churn probability.
- **Tenure** – Newer customers might churn faster, or very old customers might become dormant.
- **Country_Germany** – Specific country dynamics can influence churn.

---

## Actionable Business Recommendations

- **Targeted Re-engagement Campaigns**: Focus on customers with high Recency. Personalized offers or loyalty program reminders can encourage return visits.
- **Value-Based Retention**: Implement strategies for low Monetary value customers, such as bundled offers or loyalty discounts to increase their average basket size.
- **Onboarding for New Customers**: Pay close attention to customers with low Tenure (newer customers) to ensure positive initial experiences.
- **Geographic Specific Campaigns**: Analyze churn patterns in specific countries (e.g., Country_Germany) to tailor retention efforts.
- **Proactive Customer Service**: Use predictions to initiate outreach from support teams for high-risk customers.

---

## How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/YourUsername/Retail-Customer-Churn-Analysis.git
cd Retail-Customer-Churn-Analysis
```

## Setup Instructions

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
.env\Scriptsctivate

# On macOS/Linux:
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Download Raw Data

1. Go to the Kaggle dataset: **Online Retail II UCI**  
2. Download the `online_retail_II.zip` file.  
3. Extract `online_retail_II.xlsx` and place it in the `data/raw/` directory.

---

### Run Jupyter Notebooks (Sequential Order)

Launch Jupyter Notebook or JupyterLab from the project root directory:

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and run the notebooks in the following order:

1. `eda.ipynb`
2. `feature_engineering.ipynb`
3. `model_training_and_evaluation.ipynb`
4. `dashboard_insights.ipynb`  
   *(This will create `customer_churn_predictions.csv`)*

---

### Run the Streamlit Dashboard

Open your terminal in the project’s root directory and run:

```bash
streamlit run src/visualization_app.py
```

This will open the interactive dashboard in your web browser.

---

## Project Structure

```
Retail-Customer-Churn-Analysis/
├── .gitignore               # Specifies intentionally untracked files to ignore.
├── README.md                # This file: Project description, setup, usage.
├── requirements.txt         # Lists all Python package dependencies.
├── notebooks/               # Jupyter notebooks documenting the project workflow.
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_training_and_evaluation.ipynb
│   └── dashboard_insights.ipynb
├── src/                     # Modular Python scripts for reusable functions and Streamlit app.
│   └── visualization_app.py # The Streamlit interactive dashboard.
├── models/                  # Directory to store trained machine learning models (e.g., best_churn_model.joblib).
├── data/                    # Directory for storing raw and processed datasets.
│   ├── raw/                 # Contains the original 'online_retail_II.csv' dataset.
│   └── processed/           # Contains cleaned, feature-engineered, and predicted datasets.
├── documentation/           # Detailed documentation for each project phase.
    ├── phase1.md
    ├── phase2.md
    ├── phase3.md
    ├── phase4.md
    └── phase5.md
```

---

## Future Work / Stretch Goals

- **Advanced Feature Engineering**: Explore complex features like product categories purchased, average time between purchases, or sentiment from customer reviews (if available).
- **More Sophisticated Modeling**: Try models such as XGBoost, LightGBM, or CatBoost.
- **Model Deployment**: Simulate deploying the model as an API (e.g., Flask/FastAPI) for real-time predictions.
- **Customer Segmentation**: Use clustering for deeper customer insights and targeted strategies.
- **A/B Testing Framework**: Design a framework for testing different retention strategies.
- **Time Series Forecasting**: Use customer activity patterns to forecast future engagement.
- **Streamlit Enhancements**: Add filters, drill-downs, and segmented views for better interactivity.

---

## Contact

- **Your Name**: Sanjay Krishna MV  
- **GitHub**: https://github.com/SANJAY-KRISHNA-MV  
- **LinkedIn**: https://www.linkedin.com/in/sanjay-krishna-mv/
- **Email**: sanjaymvkrishna@gmail.com