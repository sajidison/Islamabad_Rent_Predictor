import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# 1. Load the model safely
@st.cache_resource
def load_model():
    # This looks in the same folder as this script
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        st.error(f"File NOT found at {model_path}. Check GitHub file name!")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# 2. Setup Features
all_features = ['Area', 'Bedrooms', 'Baths', 'Location_B-17', 'Location_Bahria', 'Location_Bani', 'Location_Bhara', 'Location_CBR', 'Location_Capital', 'Location_Chak', 'Location_Chatha', 'Location_Constitution', 'Location_D-12', 'Location_D-17', 'Location_DHA', 'Location_E-11', 'Location_E-16', 'Location_E-17', 'Location_E-18', 'Location_E-7', 'Location_Emaar', 'Location_F-10', 'Location_F-11', 'Location_F-15', 'Location_F-17', 'Location_F-6', 'Location_F-7', 'Location_F-8', 'Location_FECHS', 'Location_Faisal', 'Location_G-10', 'Location_G-11', 'Location_G-12', 'Location_G-13', 'Location_G-14', 'Location_G-15', 'Location_G-16', 'Location_G-6', 'Location_G-7', 'Location_G-8', 'Location_G-9', 'Location_Ghauri', 'Location_Green', 'Location_Gulberg', 'Location_Gulshan-e-Khudadad', 'Location_H-13', 'Location_I-10', 'Location_I-11', 'Location_I-13', 'Location_I-14', 'Location_I-8', 'Location_I-9', 'Location_Khanna', 'Location_Korang', 'Location_Kuri', 'Location_Lehtarar', 'Location_Margalla', 'Location_Meherban', 'Location_Mumtaz', 'Location_National', 'Location_Naval', 'Location_PWD', 'Location_Pakistan', 'Location_Park', 'Location_Police', 'Location_Shah', 'Location_Shehzad', 'Location_Soan', 'Location_Taramrri', 'Location_Tarlai', 'Location_Tarnol', 'Location_Top', 'Location_University', 'Location_Zaraj']

st.title("🇵🇰 Islamabad House Price Predictor")

# 3. User Inputs
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (Marla)", min_value=1.0, value=5.0)
    bedroom = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
with col2:
    bathroom = st.number_input("Bathrooms", min_value=1, max_value=20, value=3)
    loc_options = [loc.replace("Location_", "") for loc in all_features if "Location_" in loc]
    selected_loc = st.selectbox("Select Location", sorted(loc_options))

# 4. Prediction Logic
if st.button("Predict Price", use_container_width=True):
    # Create empty row
    input_df = pd.DataFrame(0.0, index=[0], columns=all_features)
    
    # Fill values
    input_df['Area'] = area
    input_df['Bedrooms'] = bedroom
    input_df['Baths'] = bathroom
    
    loc_col = f"Location_{selected_loc}"
    if loc_col in input_df.columns:
        input_df[loc_col] = 1.0

    # Execute Prediction
    raw_pred = model.predict(input_df)[0]
    
    # Handle Log Conversion (only if the model outputs log values)
    final_price = np.expm1(raw_pred) if raw_pred < 50 else raw_pred
    
    st.success(f"### Estimated Price: PKR {final_price:,.0f} Lakh")

