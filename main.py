import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

# Load the trained model
pickle_in = open("model_poly.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Welcome ALL"

def predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    # Convert inputs to float
    inputs = [float(industrial_risk), float(management_risk), float(financial_flexibility), float(credibility), float(competitiveness), float(operating_risk)]
    prediction = classifier.predict([inputs])
    
    # Map numeric prediction to human-readable label
    label = "Bankruptcy" if prediction[0] == 0 else "Not Bankruptcy"
    
    return label

def main():
    st.title("Bankruptcy Detector")
    html_temp = """
    <div style="background-color: tomato;padding: 10px">
    <h2 style="color: white;text-align: center;">Streamlit Bankruptcy Detector ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields
    industrial_risk = st.text_input("Industrial Risk", "Type Here")
    management_risk = st.text_input("Management Risk", "Type Here")
    financial_flexibility = st.text_input("Financial Flexibility", "Type Here")
    credibility = st.text_input("Credibility", "Type Here")
    competitiveness = st.text_input("Competitiveness", "Type Here")
    operating_risk = st.text_input("Operating Risk", "Type Here")
    
    result = ""
    
    # Predict button
    if st.button("Predict"):
        try:
            result = predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
            st.success(f"The prediction is: {result}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # About section
    if st.button("About"):
        st.text("This application predicts bankruptcy based on the input features.")
        st.text("Built with Streamlit")
        st.text("Model developed using machine learning techniques.")

if __name__ == '__main__':
    main()