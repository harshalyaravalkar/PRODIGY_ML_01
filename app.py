import streamlit as st
import pickle
import numpy as np

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# page settings
st.set_page_config(page_title="House Price Prediction")

# title
st.title("🏠 House Price Prediction")

st.write(
    "Predict house prices using square footage, bedrooms, and bathrooms."
)

# inputs
area = st.slider(
    "Square Footage",
    500,
    5000,
    1500
)

bedrooms = st.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.slider(
    "Bathrooms",
    1,
    10,
    2
)

quality = st.slider(
    "Overall Quality",
    1,
    10,
    5
)

garage = st.slider(
    "Garage Capacity",
    0,
    5,
    2
)

year = st.slider(
    "Year Built",
    1900,
    2025,
    2000
)

# prediction
if st.button("Predict Price"):

    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        quality,
        garage,
        year
    ]])

    prediction = model.predict(features)[0]

    st.success(
        f"Estimated House Price: $ {prediction:,.0f}"
    )