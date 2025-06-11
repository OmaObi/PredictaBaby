import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    """
    Simulate paediatric respiratory data with stronger
    feature–outcome relationships for model training.
    Returns a DataFrame with 11 columns.
    """
    np.random.seed(random_state)

    # Basic demographics
    age = np.random.randint(0, 61, size=n_samples)                 # months
    gender = np.random.choice([0, 1], size=n_samples)              # 0 = F, 1 = M

    # Symptoms & history
    wheeze_counts = np.random.poisson(lam=1.5, size=n_samples)     # 0-10
    sob = (wheeze_counts > np.random.randint(1, 4, size=n_samples)).astype(int)
    fam_asthma = np.random.binomial(1, 0.2, size=n_samples)        # 20 %
    eczema = np.random.binomial(1, 0.15, size=n_samples)           # 15 %

    # Vitals
    pulse = np.clip(np.random.normal(120, 15, size=n_samples), 80, 200)
    resp_rate = np.clip(np.random.normal(30, 5, size=n_samples), 15, 60)
    temp = np.clip(np.random.normal(37, 0.5, size=n_samples), 35, 39)
    spo2 = np.clip(np.random.normal(96, 2, size=n_samples), 85, 100)

    # ---------- logistic risk model ----------
    # Intercept
    logits = -2.0

    # Add weighted feature contributions
    logits += 1.2 * (wheeze_counts >= 3).astype(int)
    logits += 1.0 * sob
    logits += 0.8 * fam_asthma
    logits += 0.5 * eczema
    logits += 0.08 * (95 - spo2)               # each % below 95 raises log-odds
    logits += 0.06 * np.maximum(resp_rate - 30, 0)

    # Convert log-odds to probability
    prob = 1 / (1 + np.exp(-logits))

    # Binary outcome
    asthma = np.random.binomial(1, prob)

    return pd.DataFrame({
        "age_months": age,
        "gender": gender,
        "wheeze_counts": wheeze_counts,
        "shortness_breath": sob,
        "family_asthma": fam_asthma,
        "eczema": eczema,
        "pulse_rate": pulse,
        "respiratory_rate": resp_rate,
        "temperature": temp,
        "spo2": spo2,
        "asthma": asthma
    })
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

# Sort feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Show only top 5
top_n = 5
top_features = importance_df.head(top_n)

# === Feature Importance: Only Top 5 Shown ===
st.write("### 🌟 Most Important Features")
fig, ax = plt.subplots(figsize=(4, 2))
ax.barh(top_features["Feature"], top_features["Importance"], color='salmon')
ax.set_xlabel("Importance")
ax.invert_yaxis()
st.pyplot(fig)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

st.write("## 🧪 Model Evaluation")
st.write(f"✅ Accuracy: *{accuracy_score(y_test, y_pred):.2f}*")
st.write(f"✅ ROC AUC: *{roc_auc_score(y_test, y_prob):.2f}*")

# 3. Streamlit UI
st.write("---")
st.header("🩺 Predict Asthma Risk")

st.sidebar.header("🩺 Enter Patient Data")

age_months = st.sidebar.number_input('Age (months)', 0, 60, 24)
gender_input = st.sidebar.selectbox('Gender', ['Female', 'Male'])
wheeze_counts = st.sidebar.slider('Wheezing episodes/year', 0, 10, 2)
shortness_breath = st.sidebar.selectbox('Shortness of breath', ['No', 'Yes'])
family_asthma = st.sidebar.selectbox('Family history of asthma', ['No', 'Yes'])
eczema = st.sidebar.selectbox('Eczema/allergies', ['No', 'Yes'])
pulse_rate = st.sidebar.number_input('Pulse rate (bpm)', 80, 200, 120)
respiratory_rate = st.sidebar.number_input('Respiratory rate', 15, 60, 30)
temperature = st.sidebar.number_input('Temperature (°C)', 35.0, 39.0, 37.0)
spo2 = st.sidebar.number_input('SpO₂ (%)', 85.0, 100.0, 96.0)

submitted = st.sidebar.button("🔍 Predict Risk")

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
import base64
import datetime

if submitted:
    st.markdown("### 📄 Downloadable Report")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(200, 10, txt="PredictaBaby Asthma Risk Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt="Smart Insights for Little Lungs", ln=True, align='C')
    pdf.ln(10)

    # Timestamp
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Generated on: {now}", ln=True)
    pdf.ln(5)

    # Input Summary Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Inputs:", ln=True)
    pdf.set_font("Arial", '', 12)

    for k, v in input_dict.items():
        key_nice = k.replace('_', ' ').capitalize()
        pdf.cell(200, 8, txt=f"{key_nice}: {v}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    risk_text = f"{risk_prob * 100:.1f}%"
    pdf.cell(200, 10, txt=f"Predicted Asthma Risk: {risk_text}", ln=True, fill=True)

    if risk_prob > 0.5:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(200, 10, txt="High Risk: Consider clinical follow-up.", ln=True)
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(200, 10, txt="Low Risk: Routine monitoring advised.", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Thank you for using PredictaBaby.", ln=True, align='C')

    filename = "PredictaBaby_Asthma_Report.pdf"
    pdf.output(filename)

    with open(filename, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{filename}">📥 Click here to download your report</a>'
        st.markdown(href, unsafe_allow_html=True)

# Run with: streamlit run Trialweb.py
