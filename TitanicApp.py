import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("titanic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Titanic Survival Predictor 🚢")
st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_column_width=True)

st.write("Please enter passenger details to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Embarked Port (C = Cherbourg, Q = Queenstown, S = Southampton)", ["C", "Q", "S"])

# Convert inputs
sex = 0 if sex == "Male" else 1
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prepare input for prediction
input_data = np.array([[pclass, sex, age, fare, embarked]])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data_scaled)[0]
    if prediction == 1:
        st.success("Survived! 🎉")
        st.image("https://media.giphy.com/media/l2JehQ2GitHGdVG9y/giphy.gif", use_column_width=True)
    else:
        st.error("Did not survive 😢")
        st.image("https://media.giphy.com/media/3o6Zt6KHxJTbXCnEic/giphy.gif", use_column_width=True)