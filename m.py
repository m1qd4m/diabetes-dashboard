import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import base64

# Load CSV
df = pd.read_csv("diabetes.csv")

# Update column names
df.columns = [col.strip().replace(" ", "_") for col in df.columns]
df['Outcome'] = df['Outcome'].replace({1: 'Diabetic', 0: 'Non-Diabetic'})

# Sidebar
st.sidebar.title("🔍 Filter Patient Data")
age_range = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (20, 60))
glucose_range = st.sidebar.slider("Glucose Level Range", 0, 400, (70, 150))
bp_range = st.sidebar.slider("Blood Pressure Range", 0, 300, (50, 120))

# Filter data
df_filtered = df[
    (df['Age'].between(*age_range)) &
    (df['Glucose'].between(*glucose_range)) &
    (df['BloodPressure'].between(*bp_range))
]

# App Title
st.title("🩺 Medical Data Dashboard")
st.markdown("**App by Miqdaam** - Interactive Dashboard for Diabetes Analysis")

# Top Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df_filtered))
col2.metric("Average Age", round(df_filtered['Age'].mean(), 1))
col3.metric("Average Glucose", round(df_filtered['Glucose'].mean(), 1))

# Data Preview
st.subheader("📊 Filtered Data")
st.dataframe(df_filtered)

# Download filtered CSV
csv = df_filtered.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.markdown(f'<a href="data:file/csv;base64,{b64}" download="filtered_diabetes_data.csv">📥 Download Filtered CSV</a>', unsafe_allow_html=True)

# Altair Chart
st.subheader("📈 BMI vs Age")
alt_chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
    x=alt.X('Age', scale=alt.Scale(zero=False)),
    y=alt.Y('BMI', scale=alt.Scale(zero=False)),
    color='Outcome',
    tooltip=['Age', 'BMI', 'Outcome']
).interactive().properties(width=700, height=400)
st.altair_chart(alt_chart)

# Plotly Chart - Glucose distribution
st.subheader("📊 Glucose Level Distribution")
fig = px.histogram(df_filtered, x="Glucose", color="Outcome", nbins=30, barmode='overlay')
fig.update_layout(
    plot_bgcolor='#0e1117',
    paper_bgcolor='#0e1117',
    font=dict(color='white')
)
st.plotly_chart(fig)

# Prediction Section
st.subheader("🧪 Predict Diabetes")
preg = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose", 0, 400, 120)
bp = st.slider("Blood Pressure", 0, 300, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 21, 100, 33)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])[0]

# Show prediction
st.markdown("### 🎯 Prediction Result:")
color = "red" if pred == "Diabetic" else "green"
st.markdown(f"<h2 style='color:{color}'>{pred}</h2>", unsafe_allow_html=True)

# Classification Report
st.subheader("📋 Model Evaluation")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
acc = accuracy_score(y_test, y_pred)
st.text(f"Accuracy: {round(acc * 100, 2)}%")

st.dataframe(pd.DataFrame(report).transpose())

# Feature Importance
st.subheader("📌 Feature Importance")
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots()
feat_imp.sort_values().plot(kind='barh', color='skyblue', ax=ax2)
ax2.set_facecolor('#0e1117')
fig2.patch.set_facecolor('#0e1117')
ax2.tick_params(colors='white')
ax2.set_title("Feature Importance", color='white')
st.pyplot(fig2)

# In-App Report
st.subheader("📘 About This Dashboard")
st.markdown("""
This interactive dashboard allows you to explore and predict diabetes risk using medical data.
- **Filter** patients by Age, Glucose, and Blood Pressure.
- **View** key charts like BMI vs Age and Glucose Distribution.
- **Predict** diabetes likelihood using machine learning.
- **Understand** which features affect the outcome most.
- **Download** filtered CSVs and see model performance metrics.

> Powered by Random Forest Classifier and Streamlit. Built by **Miqdaam**.
""")
