import streamlit as st
import pandas as pd
import joblib

model = joblib.load("machine_failure_xgb.pkl")

st.set_page_config(page_title="Machine Failure Predictor", page_icon="⚙️")
st.title("⚙️ Machine Failure Prediction App")

col1, col2 = st.columns(2)
with col1:
    type_ = st.selectbox("Product Type", ["L", "M", "H"])
    air_temp = st.number_input("Air Temperature [K]", 295.0, 305.0, 298.1)
    process_temp = st.number_input("Process Temperature [K]", 305.0, 315.0, 308.6)
with col2:
    rpm = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
    torque = st.number_input("Torque [Nm]", 0.0, 80.0, 40.0)
    tool_wear = st.number_input("Tool wear [min]", 0, 300, 50)

if st.button("🔍 Predict"):
    input_df = pd.DataFrame({
        "Type": [type_],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
    })
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.error(f"⚠️ Failure predicted ({proba*100:.1f}% confidence)")
    else:
        st.success(f"✅ No failure ({proba*100:.1f}% failure probability)")
