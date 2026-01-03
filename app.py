import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the 72.71% Accuracy Brain
with open('laptop_model_final.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    le_brand = data['le_brand']

st.set_page_config(page_title="AI Price Predictor", page_icon="💻")
st.title("🚀 Chrome650 AI Price Predictor")
st.write("Professional Market Intelligence Tool | V12 Engine (72% Accuracy)")

# 1. User Inputs
brand = st.selectbox("Select Brand", sorted(le_brand.classes_))

col1, col2 = st.columns(2)
with col1:
    cpu_tier = st.select_slider("CPU Tier (1=Basic, 10=Extreme)", options=[2, 3, 5, 7, 9, 10], value=5)
    gpu_tier = st.select_slider("GPU Tier (1=Integrated, 10=Top Gaming)", options=[1, 2, 4, 6, 7, 8, 9, 10], value=4)
with col2:
    ram = st.select_slider("System RAM (GB)", options=[8, 12, 16, 24, 32, 64], value=16)
    ssd = st.select_slider("Storage (GB)", options=[256, 512, 1024, 2048], value=512)

st.divider()
is_apple = st.toggle("Is it an Apple MacBook?")
is_work = st.toggle("Is it a Workstation? (ZBook/Precision/P16v)")
is_oled = st.toggle("Does it have an OLED Display?")

# 2. Prediction Logic
if st.button("Calculate Market Value"):
    # Encode inputs to match training data
    brand_enc = le_brand.transform([brand])[0]
    apple_val = 1 if is_apple else 0
    work_val = 1 if is_work else 0
    oled_val = 1 if is_oled else 0
    
    # Interaction Feature: Power_Score = (CPU + GPU) * RAM
    power_score = (cpu_tier + gpu_tier) * ram
    
    input_df = pd.DataFrame([[brand_enc, cpu_tier, gpu_tier, ram, ssd, apple_val, work_val, oled_val, power_score]], 
                            columns=data['features'])
    
    prediction = model.predict(input_df)[0]
    
    st.subheader(f"Estimated Market Value: ₹{prediction:,.2f}")
    st.caption("Disclaimer: Prices based on current Amazon.in listing trends for new products.")
    st.balloons()