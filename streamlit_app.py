import streamlit as st
import numpy as np
import tensorflow as tf
import joblib # for loading your scaler

# 1. Load your trained model and MinMaxScaler
model = tf.keras.models.load_model('models/best_fnn_model.h5')
scaler = joblib.load('models/scaler.pkl')

st.title("🇮🇳 Indian IPO Profitability Predictor")
st.write("Enter pre-listing metrics to evaluate listing day confidence.")

# 2. Build User Input Fields
issue_size = st.number_input("Issue Size (in Crores)", min_value=10, max_value=20000, value=500)
qib_sub = st.slider("QIB Subscription Multiplier (Times)", 0.0, 200.0, 5.0)
hni_sub = st.slider("HNI Subscription Multiplier (Times)", 0.0, 200.0, 3.0)
rii_sub = st.slider("RII Subscription Multiplier (Times)", 0.0, 200.0, 2.0)
offer_price = st.number_input("Offer Price (INR)", min_value=10, max_value=5000, value=250)

# 3. Handle Live Inference
if st.button("Calculate Probability"):
    raw_features = np.array([[issue_size, qib_sub, hni_sub, rii_sub, offer_price]])
    scaled_features = scaler.transform(raw_features)
    
    prob = model.predict(scaled_features)[0][0]
    
    st.subheader(f"Listing Profit Probability: {prob * 100:.2f}%")
    if prob > 0.50:
        st.success("Verdict: High probability listing gain target!")
    else:
        st.error("Verdict: Market metrics indicate higher capital risk.")