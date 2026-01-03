import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the 72.71% Accuracy Brain
try:
    with open('laptop_model_final.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        le_brand = data['le_brand']
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'laptop_model_final.pkl' is in the repository.")

st.set_page_config(page_title="ValuWise AI", page_icon="💻")
st.title("🚀 ValuWise: Professional Laptop Pricing AI")
st.write("Day 3: Production Engine (72.71% Accuracy) | SIH 2025 Research Project")

# 1. User Inputs
brand = st.selectbox("Select Brand", sorted(le_brand.classes_))

col1, col2 = st.columns(2)
with col1:
    # CPU Scores: i3=3, i5=5, i7=7, i9=9, Ultra 9=10
    cpu_tier = st.select_slider("CPU Tier (1=Basic, 10=Extreme)", 
                               options=[2, 3, 4, 5, 6, 7, 8, 9, 10], value=5)
    
    # GPU Scores: Integrated=1, 3050=4, 4060=7, 4090=10
    gpu_tier = st.select_slider("GPU Tier (1=Integrated, 10=Top Gaming)", 
                               options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=4)
with col2:
    ram = st.select_slider("System RAM (GB)", options=[8, 12, 16, 24, 32, 64], value=16)
    ssd = st.select_slider("Storage (GB)", options=[256, 512, 1024, 2048], value=512)

st.divider()
col3, col4, col5 = st.columns(3)
with col3:
    is_apple = st.toggle("Apple MacBook?")
with col4:
    is_work = st.toggle("Workstation?")
with col5:
    is_oled = st.toggle("OLED Display?")

# 2. Market Value Calculation
if st.button("Calculate Market Value"):
    # Encode inputs
    brand_enc = le_brand.transform([brand])[0]
    apple_val = 1 if is_apple else 0
    work_val = 1 if is_work else 0
    oled_val = 1 if is_oled else 0
    
    # CRITICAL: Interaction Feature (Power_Score)
    # This logic explains the exponential price jump of gaming/pro laptops
    power_score = (cpu_tier + gpu_tier) * ram
    
    # Build input matching the 72% Model's requirements
    input_df = pd.DataFrame([[brand_enc, cpu_tier, gpu_tier, ram, ssd, apple_val, work_val, oled_val, power_score]], 
                            columns=data['features'])
    
    prediction = model.predict(input_df)[0]
    
    # Final Result Display
    st.success(f"### Estimated Market Value: ₹{prediction:,.2f}")
    st.info("AI Logic: Valuation adjusted for brand prestige, component synergy (Power Score), and display technology.")
    st.balloons()
