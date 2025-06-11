import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

st.set_page_config(page_title="PredictaBaby", layout="centered")

st.markdown("""👶🏽 
# PredictaBaby
### "Smart Insights for Little Lungs."
""")

st.markdown("""
<style>
body {
    background-image: url('https://em-content.zobj.net/thumbs/120/apple/325/baby_1f476.png');
    background-size: 60px;
    background-repeat: repeat;
}
</style>
""", unsafe_allow_html=True)

# 1. Simulate synthetic dataset based on clinical features and API criteria
def simulate_asthma_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    # Age in months (0-60 for 0-5 years)
    age = np.random.randint(0, 61, size=n_samples)
    # Gender binary
    gender = np.random.choice([0, 1], size=n_samples)  # 0=Female, 1=Male
    # Wheezing episodes per year (0-10)
    wheeze_counts = np.random.poisson(lam=1.5, size=n_samples)
    # Shortness of breath: probability increases with wheeze_counts
    sob = (wheeze_counts > np.random.randint(1, 4, size=n_samples)).astype(int)
    # Family history of asthma (20% prevalence)
    fam_asthma = np.random.binomial(1, 0.2, size=n_samples)
    # Eczema or allergies (15%)
    eczema = np.random.binomial(1, 0.15, size=n_samples)
    # Vitals: pulse rate ~ N(120, 15), resp_rate ~ N(30,5), temp ~ N(37,0.5), SpO2 ~ N(96,2)
    pulse = np.clip(np.random.normal(120, 15, size=n_samples), 80, 200)
    resp_rate = np.clip(np.random.normal(30, 5, size=n_samples), 15, 60)
    temp = np.clip(np.random.normal(37, 0.5, size=n_samples), 35, 39)
    spo2 = np.clip(np.random.normal(96, 2, size=n_samples), 85, 100)
    # Risk score probability: logistic of weighted sum
    logits = (
        0.5 * (wheeze_counts >= 3).astype(int) +
        0.4 * sob +
        0.6 * fam_asthma +
        0.3 * eczema +
        -0.01 * (spo2 - 95) +
        0.01 * (resp_rate - 30)
    )
    prob = 1 / (1 + np.exp(-logits))
    asthma = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'age_months': age,
        'gender': gender,
        'wheeze_counts': wheeze_counts,
        'shortness_breath': sob,
        'family_asthma': fam_asthma,
        'eczema': eczema,
        'pulse_rate': pulse,
        'respiratory_rate': resp_rate,
        'temperature': temp,
        'spo2': spo2,
        'asthma': asthma
    })
    return df

# Generate data
data = simulate_asthma_data()

# Split
X = data.drop('asthma', axis=1)
y = data['asthma']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Get feature importances
feature_importance = model.feature_importances_
feature_names = X.columns

# Sort features by importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=True)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

st.write("## Model Performance")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"ROC AUC: {roc_auc_score(y_test, y_prob):.2f}")
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# === Feature Importance Chart ===
st.write("### 🔍 Feature Importance")
st.bar_chart(importance_df.set_index("Feature"))

# 3. Streamlit UI
st.write("---")
st.header("🩺 Predict Asthma Risk")

with st.form("asthma_form"):
    st.subheader("Enter Patient Data")

    col1, col2 = st.columns(2)

    with col1:
        age_months = st.number_input('Age (months)', 0, 60, 24)
        gender_input = st.selectbox('Gender', ['Female', 'Male'])
        wheeze_counts = st.slider('Wheezing episodes/year', 0, 10, 2)
        shortness_breath = st.selectbox('Shortness of breath', ['No', 'Yes'])
        family_asthma = st.selectbox('Family history of asthma', ['No', 'Yes'])

    with col2:
        eczema = st.selectbox('Eczema/allergies', ['No', 'Yes'])
        pulse_rate = st.number_input('Pulse rate (bpm)', 80, 200, 120)
        respiratory_rate = st.number_input('Respiratory rate', 15, 60, 30)
        temperature = st.number_input('Temperature (°C)', 35.0, 39.0, 37.0)
        spo2 = st.number_input('SpO₂ (%)', 85.0, 100.0, 96.0)

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    # Convert inputs to the expected format
    gender = 1 if gender_input == 'Male' else 0
    sob = 1 if shortness_breath == 'Yes' else 0
    fam = 1 if family_asthma == 'Yes' else 0
    eczema_val = 1 if eczema == 'Yes' else 0

    input_dict = {
        'age_months': age_months,
        'gender': gender,
        'wheeze_counts': wheeze_counts,
        'shortness_breath': sob,
        'family_asthma': fam,
        'eczema': eczema_val,
        'pulse_rate': pulse_rate,
        'respiratory_rate': respiratory_rate,
        'temperature': temperature,
        'spo2': spo2
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    risk_prob = model.predict_proba(input_scaled)[0, 1]

    st.subheader("🧠 Model Prediction")
    st.write(f"Likelihood of asthma: *{risk_prob * 100:.1f}%*")

    if risk_prob > 0.5:
        st.warning("High risk of asthma – consider clinical follow-up.")
    else:
        st.success("Low risk of asthma.")

from fpdf import FPDF

if submitted:
    ...
    if st.button("📄 Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="PredictaBaby Asthma Risk Report", ln=True, align='C')
        pdf.ln(10)

        for k, v in input_dict.items():
            pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)

        pdf.cell(200, 10, txt=f"Predicted Risk: {risk_prob * 100:.1f}%", ln=True)
        filename = "asthma_risk_report.pdf"
        pdf.output(filename)

        with open(filename, "rb") as file:
            st.download_button("⬇ Download Your PDF", file, file_name=filename)


# Run with: streamlit run Trialweb.py
