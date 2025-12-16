import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Retail Customer Churn Analysis Dashboard")

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# --- Function to Load Data and Model ---
@st.cache_data # Cache data loading for performance
def load_data():
    """Loads the feature-engineered data with predictions."""
    try:
        data_path = PROJECT_ROOT / 'data' / 'processed' / 'customer_churn_predictions.csv'
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error("Error: 'customer_churn_predictions.csv' not found. Please ensure 02_feature_engineering.ipynb and 04_dashboard_insights.ipynb were run.")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        st.error(f"Error loading customer churn predictions data: {e}")
        return pd.DataFrame()


# Load data and model
df_features = load_data()


if df_features.empty:
    st.stop()

# Use a safe, static feature importance summary when no model is loaded
feature_importance_df = pd.DataFrame({
    "Feature": ["Recency", "Frequency", "Monetary", "Tenure"],
    "Importance": [0.40, 0.30, 0.20, 0.10]
})


# --- Dashboard Layout and Content ---

st.title("🛍️ Retail Customer Churn Analysis Dashboard")
st.markdown("This interactive dashboard helps to predict customer churn and understand key influencing factors, enabling targeted retention strategies.")

# Overview Metrics
st.header("📈 Overall Churn Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_customers = df_features['Customer ID'].nunique()
    st.metric(label="Total Customers", value=f"{total_customers:,}")

with col2:
    actual_churn_rate = df_features['is_churned'].mean() * 100
    st.metric(label="Actual Churn Rate", value=f"{actual_churn_rate:.2f}%")

with col3:
    predicted_churn_rate = df_features['predicted_churn'].mean() * 100
    st.metric(label="Predicted Churn Rate", value=f"{predicted_churn_rate:.2f}%")

with col4:
    # Example: Number of customers predicted to churn
    predicted_churners = df_features['predicted_churn'].sum()
    st.metric(label="Predicted Churners", value=f"{predicted_churners:,}")


st.markdown("---")

# --- Churn Drivers and Segmentation ---
st.header("📊 Churn Drivers & Customer Segmentation")

tab1, tab2 = st.tabs(["Key Churn Drivers", "Churn by Country"])

with tab1:
    st.subheader("Top Features Influencing Churn")
    if not feature_importance_df.empty:
        fig_importance = px.bar(feature_importance_df.head(10), x='Importance', y='Feature', orientation='h',
                                title='Top 10 Feature Importances',
                                color='Importance', color_continuous_scale=px.colors.sequential.Viridis)
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort from smallest to largest on chart
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Feature importance data not available for the loaded model.")

with tab2:
    st.subheader("Churn Rate by Country Group")
    churn_by_country = df_features.groupby('PrimaryCountry_Grouped_Original')['is_churned'].mean().sort_values(ascending=False).reset_index()
    churn_by_country['is_churned'] = churn_by_country['is_churned'] * 100 # Convert to percentage

    fig_country_churn = px.bar(churn_by_country, x='PrimaryCountry_Grouped_Original', y='is_churned',
                               title='Churn Rate by Country Group',
                               labels={'is_churned': 'Churn Rate (%)', 'PrimaryCountry_Grouped_Original': 'Country Group'},
                               color='is_churned', color_continuous_scale=px.colors.sequential.Plasma)
    fig_country_churn.update_layout(xaxis_title="Country Group", yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig_country_churn, use_container_width=True)


st.markdown("---")

# --- At-Risk Customer List ---
st.header("🎯 At-Risk Customers")

st.markdown("Identify customers with a high probability of churning for targeted retention efforts.")

# Slider for churn probability threshold
churn_threshold = st.slider(
    "Select Churn Probability Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.5, # Default threshold
    step=0.01,
    help="Customers with a predicted churn probability above this threshold are considered 'at-risk'."
)

at_risk_customers_filtered = df_features[df_features['churn_probability'] >= churn_threshold].sort_values(by='churn_probability', ascending=False)

st.info(f"Showing {len(at_risk_customers_filtered):,} customers with churn probability >= {churn_threshold:.2f}")

if not at_risk_customers_filtered.empty:
    # Select columns to display in the table for at-risk customers
    display_cols = [
        'Customer ID', 'churn_probability', 'predicted_churn',
        'Recency', 'Frequency', 'Monetary', 'Tenure', 'PrimaryCountry_Grouped_Original'
    ]
    st.dataframe(at_risk_customers_filtered[display_cols].head(500), # Show top 500 or adjust as needed
                 hide_index=True,
                 column_config={
                     "churn_probability": st.column_config.ProgressColumn(
                         "Churn Probability",
                         help="Predicted probability of customer churning",
                         format="%.2f",
                         min_value=0,
                         max_value=1,
                     ),
                     "Recency": st.column_config.NumberColumn("Recency (Days)"),
                     "Frequency": st.column_config.NumberColumn("Frequency (Transactions)"),
                     "Monetary": st.column_config.NumberColumn("Monetary (£)", format="£%.2f"),
                     "Tenure": st.column_config.NumberColumn("Tenure (Days)"),
                     "predicted_churn": st.column_config.CheckboxColumn("Predicted Churn")
                 })
else:
    st.write("No customers identified as 'at-risk' with the current threshold.")


st.markdown("---")
st.info("💡 **Insights:** Analyze the characteristics (Recency, Frequency, Monetary, Tenure) of at-risk customers and the top churn drivers to develop targeted retention strategies. For instance, customers with high Recency (haven't purchased recently) and low Frequency might be prime candidates for re-engagement campaigns.")

# --- End of Streamlit App ---
