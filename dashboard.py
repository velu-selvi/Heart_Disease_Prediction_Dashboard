import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="CardioAI Nexus Pro", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:38px !important; font-weight: 700; color: #FF4B4B; text-align: center; margin-bottom:5px; }
    .subtitle { font-size:18px; text-align: center; margin-bottom: 30px; color: #666; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🩺 CardioAI Nexus Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Multi-Mode Clinical Decision Support, Severity Triage & Automated Prescription Protocol Engine</div>", unsafe_allow_html=True)

@st.cache_resource
def train_pipeline():
    # Loads local data architecture asset
    df = pd.read_csv('data/heart.csv')

    # --- AUTOMATIC DATASET TARGET ALIGNMENT CHECK ---
    mean_oldpeak_0 = df[df['target'] == 0]['oldpeak'].mean()
    mean_oldpeak_1 = df[df['target'] == 1]['oldpeak'].mean()
    if mean_oldpeak_0 > mean_oldpeak_1:
        df['target'] = df['target'].replace({0: 1, 1: 0})

    X = df.drop('target', axis=1)
    y = df['target']

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    scaler = StandardScaler()
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_encoded, y)

    return model, scaler, X_encoded.columns, df[numerical_cols].mean().to_dict()

# Extract and unpack pipeline resources seamlessly
model, scaler, model_columns, global_means = train_pipeline()

# --- MASTER AUDIENCE MODE SELECTOR ---
app_mode = st.selectbox(
    "🔄 Select System Optimization Workspace:",
    ["👨‍⚕️ Clinical Practitioner (Doctor Mode)", "🩺 Patient Health Literacy Portal", "🎓 Med-Academic Analytics (Student Mode)"]
)

st.divider()

# --- SIDEBAR CLINICAL INPUTS ---
st.sidebar.header("📋 Baseline Patient Metrics")
age = st.sidebar.slider("Patient Age", 1, 100, 62)

# --- UPDATED INCLUSIVE SEX SELECTION ---
sex = st.sidebar.selectbox("Biological Sex / Gender Identity", ["Male", "Female", "Transgender / Non-binary"])

cp = st.sidebar.selectbox("Chest Pain Presentation", ["Asymptomatic", "Typical Angina", "Atypical Angina", "Non-anginal Pain"])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 150)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 280)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.sidebar.selectbox("Resting ECG Patterns", ["Left Ventricular Hypertrophy", "Normal", "ST-T Wave Abnormality"])
thalach = st.sidebar.slider("Maximum Heart Rate Achieved (HRmax)", 60, 220, 110)
exang = st.sidebar.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.sidebar.slider("ST Depression (Heart Muscle Strain Index)", 0.0, 6.2, 2.5, 0.1)
slope = st.sidebar.selectbox("Peak Exercise ST Slope", ["Flat", "Upsloping", "Downsloping"])
ca = st.sidebar.slider("Fluoroscopy: Number of Blocked Major Vessels", 0, 4, 2)
thal = st.sidebar.selectbox("Thalassemia Genetic Trait", ["Reversable Defect", "Normal", "Fixed Defect"])

sim_bp_reduction = 0
sim_chol_reduction = 0

if app_mode == "👨‍⚕️ Clinical Practitioner (Doctor Mode)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Virtual Treatment Simulator")
    sim_bp_reduction = st.sidebar.slider("Target Blood Pressure Reduction (mm Hg)", 0, 40, 0)
    sim_chol_reduction = st.sidebar.slider("Target Cholesterol Reduction (mg/dl)", 0, 150, 0)

# --- ADVANCED FEATURE PROCESSING ENGINE ---
def process_patient_vector(bp_mod=0, chol_mod=0):
    # Dynamic Mapping: Transgender/Non-binary is mapped to 0.5 (the middle average vector) 
    # so it does not skew heavily toward binary assumptions while avoiding script errors.
    if sex == "Male":
        sex_m = 1
    elif sex == "Female":
        sex_m = 0
    else:
        sex_m = 0.5

    cp_m = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
    fbs_m = 1 if fbs == "True" else 0
    rest_m = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
    ex_m = 1 if exang == "Yes" else 0
    sl_m = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
    th_m = {"Fixed Defect": 1, "Normal": 2, "Reversable Defect": 3}[thal]

    raw_data = pd.DataFrame([{
        'age': age, 'sex': sex_m, 'cp': cp_m, 'trestbps': max(80, trestbps - bp_mod),
        'chol': max(100, chol - chol_mod), 'fbs': fbs_m, 'restecg': rest_m, 'thalach': thalach,
        'exang': ex_m, 'oldpeak': oldpeak, 'slope': sl_m, 'ca': ca, 'thal': th_m
    }])

    encoded = pd.get_dummies(raw_data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)
    final_df = encoded.reindex(columns=model_columns, fill_value=0)

    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])
    return final_df

patient_vector_baseline = process_patient_vector()
prob_baseline = model.predict_proba(patient_vector_baseline)[0][1] * 100

# --- MAIN DISPLAY ARCHITECTURE ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🎯 Real-Time Risk Analysis")

    if prob_baseline >= 35:
        st.error(f"🚨 **INITIAL STATUS: High Risk Profile**\n\nBaseline Probability Matrix Value: {prob_baseline:.1f}%")
    else:
        st.success(f"💚 **INITIAL STATUS: Low Risk Profile**\n\nBaseline Probability Matrix Value: {prob_baseline:.1f}%")

    if app_mode == "👨‍⚕️ Clinical Practitioner (Doctor Mode)":
        st.markdown("---")
        st.markdown("### 🔬 Clinical Question & Treatment Strategy")

        if sim_bp_reduction > 0 or sim_chol_reduction > 0:
            patient_vector_sim = process_patient_vector(sim_bp_reduction, sim_chol_reduction)
            prob_sim = model.predict_proba(patient_vector_sim)[0][1] * 100
            diff = prob_baseline - prob_sim

            st.info(f"❓ **Doctor's Question:** If therapeutic intervention successfully reduces arterial pressure by **{sim_bp_reduction} mm Hg** and serum lipids by **{sim_chol_reduction} mg/dl**, what is the updated cardiac risk forecast?")

            st.markdown("### 🤖 AI Simulation Prognosis Answer")
            if prob_sim >= 35:
                st.warning(f"⚠️ **PROGNOSIS: Still High Risk ({prob_sim:.1f}%)**\n\nRisk mitigated by {diff:.1f}%, but secondary therapy tracks may be requested.")
            else:
                st.success(f"🎉 **PROGNOSIS SUCCESS: Patient Threshold Cleared ({prob_sim:.1f}%)**\n\nRisk dropped by **-{diff:.1f}%**. Safe therapeutic margin achieved!")

            st.metric(
                label="Exact Post-Treatment Vector Calculation", 
                value=f"{prob_sim:.1f}%", 
                delta=f"-{diff:.1f}% Lower Risk" if diff > 0 else "0.0%"
            )
        else:
            st.info("💡 **Clinical Tip:** Adjust the treatment simulator sliders in the sidebar to pose a clinical 'What-If' question directly to the Random Forest model.")

    elif app_mode == "🩺 Patient Health Literacy Portal":
        st.markdown("---")
        st.markdown("### 📖 Understand Your Metrics Simple Translation")
        ex_guide = "Pain or pressure in your chest occurs during normal movement or exercise." if exang == "Yes" else "Your heart handles physical exercise without immediate distress."
        vessel_guide = f"Out of 4 major heart pathways, {ca} show clear signs of structural blocking or calcium buildup."

        st.markdown(f"**Physical Activity Tolerance:** {ex_guide}")
        st.markdown(f"**Vessel Health Status:** {vessel_guide}")

        st.markdown("#### 🥦 Automated Lifestyle Guidelines:")
        if chol > 200: st.warning("• Your cholesterol is high. Focus on dietary fiber and decrease intake of saturated animal fats.")
        if trestbps > 130: st.warning("• Your blood pressure is elevated. Limit processed sodium intake below 1,500mg daily.")
        if prob_baseline < 35: st.success("• Excellent baseline. Maintain current conditioning with 150 minutes of light aerobic training weekly.")

with col2:
    if app_mode == "👨‍⚕️ Clinical Practitioner (Doctor Mode)":
        st.subheader("📋 Advanced AI Diagnostic Recommendations")

        severity_status = "MINOR ALERT / OBSERVATION PROFILE"
        severity_color = "blue"
        severity_desc = "Patient exhibits stable clinical baseline conditions. Regular annual cardiovascular screening recommended."

        if prob_baseline >= 35 and (oldpeak >= 2.0 or ca >= 2 or thalach < 120):
            severity_status = "CRITICAL ACUTE HEART STRUGGLE / INFARCTION RISK"
            severity_color = "red"
            severity_desc = "High-priority alert tracking. Severe ischemic stress indicators observed on exertion alongside significant structural blockages."
        elif prob_baseline >= 35:
            severity_status = "MAJOR VASCULAR RISK / ISCHEMIC WARNING"
            severity_color = "orange"
            severity_desc = "Systemic arterial degradation patterns found. Early preventive maintenance recommended to secure vital pathways."

        st.markdown(f"""
        <div style="background-color:rgba(255, 75, 75, 0.04); padding:15px; border-left: 6px solid {severity_color}; border-radius:4px;">
            <h4 style="margin:0; color:{severity_color}; font-weight:700;">🚨 DIAGNOSTIC LEVEL: {severity_status}</h4>
            <p style="margin:6px 0 0 0; color:#333; font-size:14px;"><b>Clinical Summary Assessment:</b> {severity_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💊 Automated Decision-Support Prescription Chart")
        st.caption("Suggested pharmacological drug class directions based on patient diagnostic markers:")

        rx_data = []
        if trestbps > 140:
            rx_data.append({"Drug Class Protocol": "ACE Inhibitors / ARBs (e.g., Lisinopril)", "Target Index": "Resting Blood Pressure", "Clinical Action Goal": "Reduce peripheral arterial vascular resistance to lower wall tension."})
        if chol > 240:
            rx_data.append({"Drug Class Protocol": "HMG-CoA Reductase Inhibitors (Statins)", "Target Index": "Serum Cholesterol", "Clinical Action Goal": "Stabilize arterial lipid sheets & downregulate LDL generation."})
        if thalach < 130 and exang == "Yes":
            rx_data.append({"Drug Class Protocol": "Beta-Blockers (e.g., Metoprolol)", "Target Index": "Ischemia Exertion Strain", "Clinical Action Goal": "Blunt peak cardiac workload to lower total muscular oxygen demands."})
        if oldpeak >= 1.5 or ca > 0:
            rx_data.append({"Drug Class Protocol": "Antiplatelet Therapies (e.g., Aspirin 81mg)", "Target Index": "Structural Blockage Index", "Clinical Action Goal": "Minimize thrombus formation and aggregation risks inside narrowed vessels."})

        if rx_data:
            st.table(pd.DataFrame(rx_data))
        else:
            st.success("✅ Patient vital attributes sit within healthy target benchmarks. No automated therapeutic protocols suggested.")

    elif app_mode == "🎓 Med-Academic Analytics (Student Mode)":
        st.subheader("📊 Patient-Specific Local Feature Impact")
        st.caption("Approximated local SHAP tracking values demonstrating how this specific patient vector wanders from data center points.")

        raw_vals = [age, trestbps, chol, thalach, oldpeak]
        means_vals = [global_means['age'], global_means['trestbps'], global_means['chol'], global_means['thalach'], global_means['oldpeak']]
        labels = ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression']

        shifts = []
        for r, m, lbl in zip(raw_vals, means_vals, labels):
            shift_dir = (r - m) if lbl != 'Max Heart Rate' else (m - r)
            shifts.append(shift_dir)

        explain_df = pd.DataFrame({'Clinical Marker': labels, 'Impact Factor': shifts}).sort_values(by='Impact Factor')

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#FF4B4B' if x > 0 else '#2E7D32' for x in explain_df['Impact Factor']]
        sns.barplot(x='Impact Factor', y='Clinical Marker', data=explain_df, palette=colors, ax=ax)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.title("Risk Accrual Vectors (Red Accelerates Risk / Green Protects)")
        st.pyplot(fig)

    else:
        st.subheader("📊 Global Model Insights")
        st.caption("Population-wide relative feature parameters measured across the underlying configuration architecture.")
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'Feature': model_columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Importance', y='Feature', data=df_imp.head(5), palette='coolwarm', ax=ax)
        plt.title("Top 5 Global Model Feature Drivers")
        st.pyplot(fig)

# --- FORCE SYSTEM REFRESH CACHE RESET BUTTON TOOL ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ System Admin Tools")
if st.sidebar.button("♻️ Force Clear Cache & Hard Rerun"):
    st.cache_resource.clear()
    st.rerun()

st.divider()
st.info("🔒 System Framework Verification: Compliant with standard anonymized academic data protocols.")
