# Phase 3: Data Preprocessing & Feature Engineering

## Phase Overview

This phase was dedicated to transforming the cleaned transactional data into a **customer-level analytical dataset** suitable for machine learning. Key activities included defining the observation and churn periods, calculating behavioral features (RFM), deriving tenure, labeling churn, and preparing categorical features for modeling.

---

## 1. Data Loading and Period Definition

- Loaded the cleaned dataset:  
  `data/processed/online_retail_cleaned.csv`
- Ensured `InvoiceDate` was parsed as a datetime object.

### Time Window Definitions:

- **analysis_end_date**: Maximum `InvoiceDate` in the dataset (`2011-12-09`)
- **churn_window_duration**: 90 days (3 months)
- **observation_end_date**: `analysis_end_date - churn_window_duration`

All features were computed using transactions **up to `observation_end_date`**, avoiding leakage from the churn window.

---

## 2. Feature Engineering: RFM Metrics

The **Recency, Frequency, and Monetary (RFM)** metrics were calculated for each customer using transactions only within the observation period.

- **Recency (R)**:  
  Days since the customer's last purchase relative to `observation_end_date`  
  `Recency = observation_end_date - max(InvoiceDate)`

- **Frequency (F)**:  
  Count of unique `InvoiceNo` values  
  `Frequency = COUNT(DISTINCT InvoiceNo)`

- **Monetary (M)**:  
  Total value of purchases  
  `Monetary = SUM(TotalPrice)`

These metrics were merged into a `customer_features` DataFrame with one row per customer.

---

## 3. Feature Engineering: Customer Tenure

- **Tenure**:  
  Days since the customer’s first transaction  
  `Tenure = observation_end_date - min(InvoiceDate)`

This feature reflects customer loyalty and longevity with the business.

---

## 4. Churn Label Creation (Target Variable)

Created a binary churn label `is_churned` based on activity across the observation and churn windows:

- Considered customers with purchases before `observation_end_date`
- **Churned (`is_churned = 1`)**:  
  Last purchase before `observation_end_date`  
  AND no purchases during the 90-day churn window
- **Not Churned (`is_churned = 0`)**:  
  Made at least one purchase within the churn window

Churn rate distribution was inspected to assess class imbalance.

---

## 5. Other Customer-Level Features & Categorical Encoding

- **Primary Country**:  
  Most frequent country for each customer, used as a geographic proxy.

### Country Encoding Strategy:

- Top 10 countries retained individually
- Remaining countries grouped into **'Other'**
- One-hot encoded using `pd.get_dummies()`  
  `drop_first=True` to avoid multicollinearity

---

## 6. Output

The final feature set was stored in:

```bash
data/processed/customer_churn_features.csv