import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --------------------
# Load and prepare data
# --------------------
df = pd.read_csv("diabetes.csv")
df['Outcome'] = df['Outcome'].map({0: 'Non-Diabetic', 1: 'Diabetic'})

# --------------------
# Dark Theme Styling
# --------------------
st.set_page_config(layout="wide", page_title="Medical Diabetes Dashboard", page_icon="💉")

st.markdown("""
    <style>
        body {
            color: #fff;
            background-color: #0E1117;
        }
        .main {
            background-color: #0E1117;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
        }
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------
# Sidebar Filters
# --------------------
st.sidebar.title("🔎 Filter Data")

age = st.sidebar.slider("Age", int(df.Age.min()), 100, (20, 60))
glucose = st.sidebar.slider("Glucose", 0, 400, (70, 150))
bp = st.sidebar.slider("Blood Pressure", 0, 300, (60, 140))

filtered_df = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1]) &
                 (df['Glucose'] >= glucose[0]) & (df['Glucose'] <= glucose[1]) &
                 (df['BloodPressure'] >= bp[0]) & (df['BloodPressure'] <= bp[1])]

# --------------------
# Main Title & Dataset
# --------------------
st.title("💉 Medical Diabetes Dashboard")
st.markdown("**Built by Miqdaam**")

# --------------------
# Top Metrics
# --------------------
col1, col2, col3 = st.columns(3)
col1.metric("📋 Total Records", len(filtered_df))
col2.metric("📋 Average Age", round(filtered_df['Age'].mean(), 1))
col3.metric("🧪 Avg. Glucose", round(filtered_df['Glucose'].mean(), 1))

# --------------------
# Data Table & Download
# --------------------
st.dataframe(filtered_df, use_container_width=True)

csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered CSV", data=csv, file_name="filtered_data.csv", mime='text/csv')

# --------------------
# Charts Section
# --------------------
st.subheader("📊 Visual Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_df, x="Glucose", color="Outcome", nbins=30,
                        title="Glucose Distribution by Outcome", height=450,
                        color_discrete_map={"Diabetic": "#FF4B4B", "Non-Diabetic": "#00C49A"})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    alt_chart = alt.Chart(filtered_df).mark_circle(size=80).encode(
        x=alt.X('BMI', title='BMI'),
        y=alt.Y('Age', title='Age'),
        color=alt.Color('Outcome', scale=alt.Scale(domain=["Non-Diabetic", "Diabetic"],
                                                   range=['#00C49A', '#FF4B4B']))
    ).properties(
        title="BMI vs Age Chart",
        width=600,
        height=450
    )
    st.altair_chart(alt_chart, use_container_width=True)

# --------------------
# Machine Learning Prediction
# --------------------
st.subheader("🤖 Predict Diabetes")

with st.form("predict_form"):
    preg = st.number_input("Pregnancies (For Females)", 0, 20)
    glucose = st.slider("Glucose Level", 0, 400, 120)
    bp = st.slider("Blood Pressure", 0, 300, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age_input = st.slider("Age", 10, 100, 30)

    submit = st.form_submit_button("Predict")

    if submit:
        X = df.drop(columns="Outcome")
        y = df["Outcome"].map({'Non-Diabetic': 0, 'Diabetic': 1})

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        prediction = model.predict([[preg, glucose, bp, skin, insulin, bmi, dpf, age_input]])[0]
        pred_label = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
        pred_color = 'red' if prediction == 1 else 'green'

        st.markdown(f"<h3 style='color:{pred_color}'>Prediction: {pred_label}</h3>", unsafe_allow_html=True)

        # --------------------
        # Feature Importance
        # --------------------
        st.subheader("📌 Feature Importance")

        feat_imp = pd.Series(model.feature_importances_, index=X.columns)
        fig2, ax2 = plt.subplots()
        feat_imp.sort_values().plot(kind='barh', color='skyblue', ax=ax2)
        ax2.set_facecolor('#0e1117')
        fig2.patch.set_facecolor('#0e1117')
        ax2.tick_params(colors='white')
        ax2.set_title("Feature Importance", color='white')
        st.pyplot(fig2)

# --------------------
# Report Section
# --------------------
st.subheader("📄 Report Summary")

st.markdown("""
- The app analyzes medical data to identify diabetic patients.
- You can filter data and visualize key metrics like Glucose, Blood Pressure, Age, BMI.
- A machine learning model is trained to predict whether a person is diabetic.
- Feature importance shows which medical factors affect prediction most.
- This dashboard can help medical professionals understand trends in diabetes data.
""")

# Footer
st.markdown("<hr><center>Made with ❤️ by Miqdaam Muneeb</center>", unsafe_allow_html=True)
