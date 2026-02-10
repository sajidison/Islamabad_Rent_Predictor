

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model (ensure it was saved using TransformedTargetRegressor for best results)
model = joblib.load("model_perfect_pipeline.pkl")

# Exact feature list from your training data
all_features = ['Area', 'Bedrooms', 'Baths', 'Location_B-17', 'Location_Bahria', 'Location_Bani', 'Location_Bhara',
                'Location_CBR', 'Location_Capital', 'Location_Chak', 'Location_Chatha', 'Location_Constitution',
                'Location_D-12', 'Location_D-17', 'Location_DHA', 'Location_E-11', 'Location_E-16', 'Location_E-17',
                'Location_E-18', 'Location_E-7', 'Location_Emaar', 'Location_F-10', 'Location_F-11', 'Location_F-15',
                'Location_F-17', 'Location_F-6', 'Location_F-7', 'Location_F-8', 'Location_FECHS', 'Location_Faisal',
                'Location_G-10', 'Location_G-11', 'Location_G-12', 'Location_G-13', 'Location_G-14', 'Location_G-15',
                'Location_G-16', 'Location_G-6', 'Location_G-7', 'Location_G-8', 'Location_G-9', 'Location_Ghauri',
                'Location_Green', 'Location_Gulberg', 'Location_Gulshan-e-Khudadad', 'Location_H-13', 'Location_I-10',
                'Location_I-11', 'Location_I-13', 'Location_I-14', 'Location_I-8', 'Location_I-9', 'Location_Khanna',
                'Location_Korang', 'Location_Kuri', 'Location_Lehtarar', 'Location_Margalla', 'Location_Meherban',
                'Location_Mumtaz', 'Location_National', 'Location_Naval', 'Location_PWD', 'Location_Pakistan',
                'Location_Park', 'Location_Police', 'Location_Shah', 'Location_Shehzad', 'Location_Soan',
                'Location_Taramrri', 'Location_Tarlai', 'Location_Tarnol', 'Location_Top', 'Location_University',
                'Location_Zaraj']

st.title("Property Price Predictor")

# Input widgets
area = st.number_input("Area (Marla)", value=5.0)
bedroom = st.selectbox("Bedrooms", options=range(1, 11), index=2)
bathroom = st.selectbox("Bathrooms", options=range(1, 11), index=1)
location = st.selectbox("Location", [loc.replace("Location_", "") for loc in all_features if "Location_" in loc])

if st.button("Predict Price"):
    # Create empty DataFrame with all 0s
    input_df = pd.DataFrame(0, index=[0], columns=all_features)

    # Map inputs to DataFrame
    input_df['Area'] = area
    input_df['Bedrooms'] = bedroom
    input_df['Baths'] = bathroom

    # Set the one-hot encoded location column to 1
    loc_col = f"Location_{location}"
    if loc_col in input_df.columns:
        input_df[loc_col] = 1

    # Get prediction
    prediction = model.predict(input_df)[0]

    # If your model was NOT wrapped in TransformedTargetRegressor, use: final_price = np.expm1(prediction)
    # If it WAS wrapped, use: final_price = prediction
    final_price = np.expm1(prediction) if prediction < 50 else prediction

    st.success(f"Estimated Price: PKR {final_price:,.0f} LAKH")