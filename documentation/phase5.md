# Phase 5: Visualization & Communication

## Phase Overview

This phase focused on converting churn prediction insights into an interactive, user-friendly dashboard. The goal was to enable business stakeholders to:

- Understand customer churn behavior
- Identify at-risk customers
- Interpret the key factors driving churn
- Facilitate data-driven decisions for retention strategies

---

## 1. Tool Selection: Streamlit for Interactive Dashboard

**Why Streamlit?**

- Python-native solution
- Easily shareable from a GitHub repository
- Simple to use for building web apps

**Visualization Library Used:**  
- `Plotly Express` for interactive charts

---

## 2. Dashboard Development (`src/visualization_app.py`)

The Streamlit dashboard was developed as a Python script that performs the following functions:

### ✅ Data and Model Loading

- Loads:
  - `customer_churn_predictions.csv` (engineered features with predictions)
  - `best_churn_model.joblib` (Gradient Boosting Classifier)

- Uses:
  - `@st.cache_data` and `@st.cache_resource` decorators for efficient performance
  - Robust error handling for missing files

---

### 📊 Overall Churn Insights

Uses `st.metric` to display:

- Total Customers
- Actual Churn Rate
- Predicted Churn Rate
- Number of Predicted Churners

---

### 💡 Churn Drivers & Customer Segmentation

- **Top Features Influencing Churn**:
  - Derived from Gradient Boosting model's feature importances
  - Visualized using a horizontal bar chart (Plotly Express)
  - Helps identify key churn drivers (e.g., Recency, Frequency, Tenure)

- **Churn Rate by Country Group**:
  - Bar chart of churn rates by primary country group
  - Useful for regional strategy development

---

### 🔍 At-Risk Customers List

- **Dynamic Filtering**:
  - Slider to adjust churn probability threshold
  - Filters customers above the selected threshold

- **Interactive Table**:
  - Displays:
    - Customer ID
    - Churn Probability
    - Predicted Churn
    - RFM values
    - Tenure
    - Country

- **Enhanced Readability**:
  - Uses `st.column_config` for:
    - Progress bars for churn probability
    - Numeric formatting

---

## 3. Empowerment for End-Users

The dashboard bridges the gap between technical modeling and business decision-making by:

- 📉 Simplifying complex insights through clear visuals
- 🎯 Enabling targeted action on individual at-risk customers
- 📈 Informing overall strategy based on churn driver analysis
- 🧩 Encouraging exploration with interactive elements

---

## 4. Execution

Run the dashboard with:

```bash
streamlit run src/visualization_app.py
```