import streamlit as st
import numpy as np
import pickle


def data():
    with open('/Users/sathwikvuppala/Desktop/Capstone/saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data =  data()



regressor = data["model"]
le_smoker = data["le_smoker"]
le_region = data["le_region"]

def show_predict_page():
    st.title("Medical Insurance Cost Prediction")
    
    smoker = (
        "yes",
        "no",
    )
    
    region = (
        "northeast",
        "northwest",
        "southeast",
        "southwest",
    )
    
    age = st.slider("Age of the Person",0, 100, 25)
    bmi = st.slider("BMI of the Person",20.00, 50.00, 25.00)
    children = st.slider("Number of Children for the Person",0, 10, 2)
    smoker = st.selectbox("Does the Person Smoke ?",smoker)
    region = st.selectbox("Region of the Person ?",region)
    
    ok = st.button("Predict")
    
    if ok :
        X = np.array([[age, bmi, children, smoker, region]])
        X[:, 3] = le_smoker.transform(X[:,3])
        X[:, 4] = le_region.transform(X[:,4])
        X = X.astype(float)
        
        cost = regressor.predict(X)
        st.subheader(f"The estiated Medical Insurance in ${cost[0]:.2f}")