import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("ElasticNet_Purchase_Prediction.pkl")

# Page title
st.title("🛍️ Predict Customer Purchases 💰")
st.markdown("Enter customer details and let the algorithm guess how much they'll splash 💸")

# Categorical encodings
location_map = {"Barisal": 0, "Chittagong": 1, "Dhaka": 2, "Khulna": 3,
                "Mymensingh": 4, "Rajshahi": 5, "Rangpur": 6, "Sylhet": 7}
loyalty_map = {"Bronze": 0, "Gold": 1, "Silver": 2}

# Input fields
Items = st.selectbox("🧺 Number of Items Bought", [1, 2, 3, 4, 5, 6], 5)
Average = st.slider("💸 Average Spend per Item", 200, 1500, 500)
location = st.selectbox("📍 Customer Location", list(location_map.keys()), 0)
loyalty = st.selectbox("🏅 Loyalty Tier", list(loyalty_map.keys()), 1)
lastseen = st.slider("⏳ Days Since Last Login", 15, 155, 50)

# Convert to model-ready format
input_df = pd.DataFrame([{
    "items": Items,
    "average": Average,
    "location": location_map[location],
    "loyalty": loyalty_map[loyalty],
    "lastseen": lastseen
}])

# Show input data
st.subheader("📋 Customer Data Preview")
st.write(input_df)

# Predict button
if st.button("🔮 Predict Purchase"):
    prediction = model.predict(input_df)[0]
    st.success(f"🧾 Estimated Purchase Amount: **${prediction:,.2f}**")
    st.markdown("🛒 _Looks like this customer's wallet is warming up!_")
