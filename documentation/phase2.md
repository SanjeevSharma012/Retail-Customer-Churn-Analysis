# Phase 2: Data Acquisition & Understanding

## Phase Overview

This phase focused on acquiring the raw transactional data, performing initial data loading and cleaning, and conducting in-depth Exploratory Data Analysis (EDA). The goal was to understand the dataset’s structure, handle data quality issues, and extract preliminary insights that would inform future feature engineering and modeling. The phase concluded with a defined churn observation window, which is essential for labeling the target variable.

---

## 1. Data Acquisition

The project uses the **Online Retail II** dataset from Kaggle:  
[https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

- Source: UK-based online retail store  
- Time Period: December 2009 – December 2011  
- File: `online_retail_II.xlsx`  
- Location: Placed in the `data/raw/` directory of the project repository

---

## 2. Initial Data Loading and Inspection

The initial steps in `eda.ipynb` involved loading the Excel file into a Pandas DataFrame. Key inspection commands used:

- `df.info()`: Check data types, null values, memory usage  
- `df.head()`: Preview the first few rows  
- `df.describe()`: Summary statistics of numerical columns  
- `df.isnull().sum()`: Count of missing values per column  
- `df['Country'].value_counts()`: Distribution of customer countries

### Key Observations:

- Significant missing values in **CustomerID**
- Missing values in **Description**
- Non-positive **Quantity** and **Price** values (e.g., returns)
- `Invoice` with prefix `'C'` indicate **cancelled orders**
- `InvoiceDate` required conversion to datetime format
- Column `Price` was actually **unit price**, not `UnitPrice`

---

## 3. Basic Data Cleaning

To prepare the dataset for modeling:

- **Dropped rows** with missing `CustomerID`
- Converted `CustomerID` to integer, `InvoiceDate` to datetime
- Filled missing `Description` values with `'Unknown'`
- Removed rows where `Quantity` or `Price` ≤ 0
- Removed cancelled transactions (`Invoice` starting with `'C'`)
- Created new column:  
  `TotalPrice = Quantity * Price`
- Removed **duplicate rows**
- Saved cleaned data to:  
  `data/processed/online_retail_cleaned.csv`

---

## 4. In-depth Exploratory Data Analysis (EDA)

### Time Range Analysis

- **Date Range**: `2009-12-01` to `2011-12-09`  
- Visualized:
  - Monthly unique transactions
  - Total sales trends and seasonality

### Distribution Analysis

- Histograms for `Quantity`, `Price`, `TotalPrice`
- Distributions were **right-skewed**
- **Logarithmic scales** used for clearer visualizations

### Customer & Country Insights

- Total **unique customers** calculated
- Top 10 countries by:
  - Number of transactions
  - Total sales  
- **UK** dominates the dataset, as expected

---

## 5. Churn Definition Strategy

Since no churn label was provided, a custom churn definition was established:

- **Analysis End Date**: `2011-12-09` (latest InvoiceDate)
- **Churn Window**: 3 months  
- **Observation End Date**:  
  `analysis_end_date - 3 months`

### Churn Labeling

- **Churned (`is_churned = 1`)**:  
  Last purchase before `observation_end_date`  
  **AND** no purchase during the churn window

- **Not Churned (`is_churned = 0`)**:  
  Had activity before `observation_end_date`  
  **AND** made at least one purchase during the churn window