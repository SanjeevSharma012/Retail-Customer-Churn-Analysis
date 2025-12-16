# Project Planning & Setup

## Phase Overview

This document outlines the initial planning and setup phase for the **"Retail Customer Churn Analysis"** project. The primary goal of this phase was to define the project scope, break down the project into manageable tasks, select initial tools and libraries, and establish a clear GitHub repository structure. This foundational work ensures a structured and efficient development process.

---

## 1. Project Objectives & Scope Definition

The core objective of this project is to build a **machine learning model** capable of predicting customer churn in a retail business using a real-world transactional dataset. Beyond prediction, the project aims to identify the key factors influencing churn, enabling the development of targeted customer retention strategies.

The project is structured to showcase skills critical for a **Data Scientist** role, including:

- **Data Acquisition & Preparation**:  
  Working with a real-world, publicly available dataset (Online Retail II), demonstrating skills in loading, inspecting, and initially cleaning raw data.

- **Data Manipulation & Feature Engineering**:  
  Creating meaningful features like RFM (Recency, Frequency, Monetary) and deriving customer-level insights (e.g., customer tenure) from raw transactional data.

- **Statistical Modeling & ML Algorithms**:  
  Building and evaluating classification models.

- **Model Evaluation**:  
  Using appropriate metrics (accuracy, precision, recall, F1-score, ROC-AUC).

- **Codified Algorithms**:  
  Developing clean, modular, and well-commented Python code.

- **Visualization & Communication**:  
  Creating interactive dashboards and deriving actionable business insights.

- **Business Understanding & Problem Solving**:  
  Translating analytical findings into practical business recommendations within a retail context.

---

## Project Breakdown into Manageable Phases

The project has been segmented into the following key phases:

1. **Project Planning & Setup** *(Current Phase)*:  
   Defining scope, setting up environment, initial tool selection.

2. **Data Acquisition & Understanding**:  
   Acquiring the specified Kaggle dataset, performing initial Exploratory Data Analysis (EDA), and basic data cleaning.

3. **Data Preprocessing & Feature Engineering**:  
   Cleaning data, transforming features, and creating new predictive variables (e.g., RFM, customer tenure) from the transactional data.

4. **Modeling & Evaluation**:  
   Selecting, training, optimizing, and evaluating various machine learning churn prediction models.

5. **Visualization & Communication**:  
   Developing interactive dashboards and formulating actionable business recommendations.

6. **Documentation & Presentation**:  
   Creating comprehensive project documentation, well-commented code, and a final project report/presentation.

---

## 2. Data Source and Initial Considerations

Instead of generating hypothetical data, this project utilizes the **"Online Retail II" dataset** from Kaggle:  
[https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

### Key aspects and adjustments due to this data source:

- **Dataset Content**:  
  Includes fields like `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

- **Missing Demographic Data**:  
  No direct demographic information (age, gender, etc.). `Country` will serve as the primary geographical demographic feature.

- **Churn Definition**:  
  Churn is defined based on **customer inactivity** (Recency) over a specific period.

- **Customer Registration Date**:  
  Will be engineered using the earliest transaction date per customer.

- **Focus on Feature Engineering**:  
  Due to the lack of direct demographics, more focus will be on **behavioral features** from the transactional history (e.g., RFM).

---

## 3. Initial Tool & Library Choices

The following tools and libraries were selected to support project development, balancing industry standards and ease of use:

- **Python**: Primary programming language for all tasks.

- **pandas**: For data manipulation, cleaning, and analysis (including Excel handling).

- **numpy**: Fundamental library for numerical operations.

- **scikit-learn**: For building and evaluating ML models:
  - Classification algorithms: Logistic Regression, Random Forest, Gradient Boosting
  - Model selection tools and evaluation metrics

- **matplotlib.pyplot** & **seaborn**: For static visualizations during EDA.

- **plotly / dash / streamlit**:  
  Under consideration for interactive dashboards.  
  *Preferred: Streamlit or Dash for seamless Python integration.*

- **joblib / pickle**: For saving/loading trained models.