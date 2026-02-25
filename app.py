import streamlit as st
import pandas as pd
import joblib

# Load models
rating_model = joblib.load("rating_model.pkl")
visit_model = joblib.load("visit_model.pkl")
visit_encoder = joblib.load("visit_encoder.pkl")

df = pd.read_csv("tourism.csv")

st.title("✈️ AIML Tourism Experience Analytics")

# Dropdowns
country = st.selectbox("Country", df["Country"].unique())
city = st.selectbox("City", df["City"].unique())
attraction = st.selectbox("Attraction", df["Attraction"].unique())
year = st.selectbox("Year", sorted(df["Year"].unique()))
month = st.selectbox("Month", sorted(df["Month"].unique()))

# Encode inputs
country_val = df[df["Country"] == country]["Country"].index[0]
city_val = df[df["City"] == city]["City"].index[0]
attr_val = df[df["Attraction"] == attraction]["Attraction"].index[0]

X = [[country_val, city_val, attr_val, year, month]]

if st.button("Predict Experience"):

    rating = rating_model.predict(X)[0]
    visit = visit_model.predict(X)[0]
    visit_mode = visit_encoder.inverse_transform([visit])[0]

    st.success(f"Predicted Rating: {round(rating,2)}")
    st.success(f"Predicted Visit Mode: {visit_mode}")

    # Recommendation
    rec = df[df["Rating"] >= 4]["Attraction"].unique()
    st.subheader("Recommended Attractions")
    for r in rec:
        st.write("•", r)