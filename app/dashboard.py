import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.predict import predict_from_csv


st.set_page_config(page_title="Spot the Scam", layout="wide")

st.title("🕵️‍♀️ Spot the Scam – Job Fraud Detector")

uploaded_file = st.file_uploader("Upload a job listing CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    result = predict_from_csv(df)

    st.subheader("📋 Prediction Table")
    st.dataframe(result[['title', 'location', 'fraud_probability', 'predicted_label']])

    st.subheader("📊 Fraud Probability Histogram")
    fig1 = px.histogram(result, x="fraud_probability", nbins=50, title="Fraud Probability Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🥧 Fraud vs Genuine Pie Chart")
    pie_data = result['predicted_label'].value_counts().rename_axis('label').reset_index(name='count')
    pie_data['label'] = pie_data['label'].map({0: 'Genuine', 1: 'Fraud'})
    fig2 = px.pie(pie_data, values='count', names='label', title="Fraudulent vs Genuine")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("☁️ Word Cloud of Fraudulent Job Descriptions")
    fraud_jobs = result[result['predicted_label'] == 1]
    text = ' '.join(fraud_jobs['description'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text)
    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig_wc)
    st.subheader("📊 Fraud Probability Distribution")
    if 'fraud_probability' in result.columns:
        fig3 = px.histogram(
            result,
            x='fraud_probability',
            nbins=20,
            title="Distribution of Fraud Probabilities",
            color='predicted_label',
            color_discrete_map={0: 'green', 1: 'red'},
            labels={'fraud_probability': 'Fraud Probability'},
            opacity=0.75
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ 'fraud_probability' column not found in predictions.")
    
    st.subheader("🚨 Top 10 Most Suspicious Job Listings")
    if 'fraud_probability' in result.columns and 'title' in result.columns:
        top_suspicious = result.sort_values(by='fraud_probability', ascending=False).head(10)
        st.dataframe(top_suspicious[['title', 'company_profile', 'location', 'fraud_probability']])
    else:
        st.warning("⚠️ Required columns not found to show top suspicious listings.")
    
    st.subheader("📌 Summary Metrics")

    total_jobs = len(result)
    total_fraud = result['predicted_label'].sum()
    total_genuine = total_jobs - total_fraud
    fraud_rate = (total_fraud / total_jobs) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", total_jobs)
    col2.metric("Fraudulent Jobs", total_fraud)
    col3.metric("Genuine Jobs", total_genuine)
    col4.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")
    
    st.subheader("🔍 Explore Job Listings by Filters")

    locations = result['location'].dropna().unique()
    titles = result['title'].dropna().unique()

    selected_location = st.selectbox("Select Location", options=np.insert(locations, 0, "All"))
    selected_title = st.selectbox("Select Job Title", options=np.insert(titles, 0, "All"))

    filtered_data = result.copy()
    if selected_location != "All":
        filtered_data = filtered_data[filtered_data['location'] == selected_location]
    if selected_title != "All":
        filtered_data = filtered_data[filtered_data['title'] == selected_title]

    st.write(f"Showing {len(filtered_data)} job listings based on selected filters.")

    st.dataframe(filtered_data[['title', 'company_profile', 'location', 'fraud_probability']])

