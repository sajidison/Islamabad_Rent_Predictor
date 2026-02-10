import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Get the directory of the current script to ensure the model file is found
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "model_perfect_pipeline.pkl")

# Load the model with error handling
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file not found! Please ensure '{os.path.basename(model_path)}' is in your GitHub repo.")
    st.stop()

# Exact feature list from training
all_features = ['Area', 'Bedrooms', 'Baths', 'Location_B-17', 'Location_Bahria', 'Location_Bani', 'Location_Bhara', 'Location_CBR', 'Location_Capital', 'Location_Chak', 'Location_Chatha', 'Location_Constitution', 'Location_D-12', 'Location_D-17', 'Location_DHA', 'Location_E-11', 'Location_E-16', 'Location_E-17', 'Location_E-18', 'Location_E-7', 'Location_Emaar', 'Location_F-10', 'Location_F-11', 'Location_F-15', 'Location_F-17', 'Location_F-6', 'Location_F-7', 'Location_F-8', 'Location_FECHS', 'Location_Faisal', 'Location_G-10', 'Location_G-11', 'Location_G-12', 'Location_G-13', 'Location_G-14', 'Location_G-15', 'Location_G-16', 'Location_G-6', 'Location_G-7', 'Location_G-8', 'Location_G-9', 'Location_Ghauri', 'Location_Green', 'Location_Gulberg', 'Location_Gulshan-e-Khudadad', 'Location_H-13', 'Location_I-10', 'Location_I-11', 'Location_I-13', 'Location_I-14', 'Location_I-8', 'Location_I-9', 'Location_Khanna', 'Location_Korang', 'Location_Kuri', 'Location_Lehtarar', 'Location_Margalla', 'Location_Meherban', 'Location_Mumtaz', 'Location_National', 'Location_Naval', 'Location_PWD', 'Location_Pakistan', 'Location_Park', 'Location_Police', 'Location_Shah', 'Location_Shehzad', 'Location_Soan', 'Location_Taramrri', 'Location_Tarlai', 'Location_Tarnol', 'Location_Top', 'Location_University', 'Location_Zaraj']

st.set_page_config(page_title="Islamabad Property Predictor")
st.title("🏡 Islamabad Property Price Predictor")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (Marla)", min_value=1.0, value=5.0, step=0.5)
    bedroom = st.selectbox("Bedrooms", options=list(range(1, 11)), index=2)
with col2:
    bathroom = st.selectbox("Bathrooms", options=list(range(1, 11)), index=1)
    location_name = st.selectbox("Location", [loc.replace("Location_", "") for loc in all_features if "Location_" in loc])

if st.button("Predict Price", use_container_width=True):
    # Prepare input data
    input_df = pd.DataFrame(0.0, index=[0], columns=all_features)
    input_df['Area'], input_df['Bedrooms'], input_df['Baths'] = area, bedroom, bathroom
    
    loc_col = f"Location_{location_name}"
    if loc_col in input_df.columns:
        input_df[loc_col] = 1.0

    # Prediction logic
    prediction = model.predict(input_df)[0]
    
    # Check if we need to inverse log (usually log values are < 50 for house prices)
    final_price = np.expm1(prediction) if prediction < 50 else prediction

    st.balloons()
    st.success(f"### Estimated Price: PKR {final_price:,.0f}")
