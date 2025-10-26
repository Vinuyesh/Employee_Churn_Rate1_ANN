#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import streamlit as st
# from keras.models import load_model
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# # Load trained model
# # model = load_model("models/employee_exit_model.h5")
# model = load_model("models/employee_exit_model.keras")

# st.title("Bank Employee Exit Prediction 💼")
# st.write("Enter employee details to predict if they might exit the company.")

# # User inputs
# credit_score = st.number_input("Credit Score", 300, 850)
# age = st.number_input("Age", 18, 65)
# tenure = st.number_input("Tenure (years)", 0, 10)
# balance = st.number_input("Balance", 0.0, 1000000.0)
# num_of_products = st.number_input("Number of Products", 1, 10)
# has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
# is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
# estimated_salary = st.number_input("Estimated Salary", 0.0, 500000.0)

# # Convert categorical to numeric
# has_cr_card = 1 if has_cr_card == "Yes" else 0
# is_active_member = 1 if is_active_member == "Yes" else 0

# # Prediction button
# if st.button("Predict"):
#     X_new = np.array([[credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
    
#     # Optional: scale inputs the same way as training
#     scaler = StandardScaler()
#     X_new_scaled = scaler.fit_transform(X_new)  # Only for demo; ideally use same scaler as training
    
#     prediction = model.predict(X_new_scaled)
#     st.write("⚠ Employee is likely to EXIT" if prediction[0][0] > 0.5 else "✅ Employee will STAY")


# In[5]:


# import streamlit as st
# from keras.models import load_model
# import numpy as np
# import joblib

# # Load trained model and scaler
# model = load_model("models/employee_exit_model.keras")
# scaler = joblib.load("models/scaler.save")

# st.title("Bank Employee Exit Prediction 💼")
# st.write("Enter employee details to predict if they might exit the company.")

# # User inputs
# credit_score = st.number_input("Credit Score", 300, 850)
# age = st.number_input("Age", 18, 65)
# tenure = st.number_input("Tenure (years)", 0, 10)
# balance = st.number_input("Balance", 0.0, 1000000.0)
# num_of_products = st.number_input("Number of Products", 1, 10)
# has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
# is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
# estimated_salary = st.number_input("Estimated Salary", 0.0, 500000.0)
# gender = st.selectbox("Gender", ["Male", "Female"])
# geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# # Convert categorical inputs to numeric
# has_cr_card = 1 if has_cr_card == "Yes" else 0
# is_active_member = 1 if is_active_member == "Yes" else 0
# gender = 1 if gender == "Male" else 0

# # Geography one-hot encoding (matching training dummies)
# geo_germany = 1 if geography == "Germany" else 0
# geo_spain = 1 if geography == "Spain" else 0

# # Prepare input array with all 12 features
# X_new = np.array([[credit_score, gender, age, tenure, balance, num_of_products,
#                    has_cr_card, is_active_member, estimated_salary, geo_germany, geo_spain]])

# # Prediction button
# if st.button("Predict"):
#     # Scale using the same scaler as training
#     X_new_scaled = scaler.transform(X_new)
    
#     prediction = model.predict(X_new_scaled)
#     st.write("⚠ Employee is likely to EXIT" if prediction[0][0] > 0.5 else "✅ Employee will STAY")


# In[7]:


# import streamlit as st
# from keras.models import load_model
# import numpy as np
# import joblib  # to load the saved scaler

# # Load trained model and scaler
# model = load_model("models/employee_exit_model.keras")
# scaler = joblib.load("models/scaler.save")

# st.title("Bank Employee Exit Prediction 💼")
# st.write("Enter employee details to predict if they might exit the company.")

# # User inputs
# credit_score = st.number_input("Credit Score", 300, 850)
# age = st.number_input("Age", 18, 65)
# tenure = st.number_input("Tenure (years)", 0, 10)
# balance = st.number_input("Balance", 0.0, 1000000.0)
# num_of_products = st.number_input("Number of Products", 1, 10)
# has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
# is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
# estimated_salary = st.number_input("Estimated Salary", 0.0, 500000.0)

# # Convert categorical to numeric
# has_cr_card = 1 if has_cr_card == "Yes" else 0
# is_active_member = 1 if is_active_member == "Yes" else 0

# # Prediction button
# if st.button("Predict"):
#     # Make input array
#     X_new = np.array([[credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
    
#     # Use the saved scaler to scale inputs
#     X_new_scaled = scaler.transform(X_new)
    
#     # Predict
#     prediction = model.predict(X_new_scaled)
#     st.write("⚠ Employee is likely to EXIT" if prediction[0][0] > 0.5 else "✅ Employee will STAY")


# In[8]:


import streamlit as st
from keras.models import load_model
import numpy as np
import joblib

# Cache the model and scaler so they're loaded only once
@st.cache_resource
def load_model_and_scaler():
    model = load_model("models/employee_exit_model.keras")
    scaler = joblib.load("models/scaler.save")
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Bank Employee Exit Prediction 💼")
st.write("Enter employee details to predict if they might exit the company.")

# User inputs
credit_score = st.number_input("Credit Score", 300, 850)
age = st.number_input("Age", 18, 65)
tenure = st.number_input("Tenure (years)", 0, 10)
balance = st.number_input("Balance", 0.0, 1000000.0)
num_of_products = st.number_input("Number of Products", 1, 10)
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", 0.0, 500000.0)

# Convert categorical to numeric
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

if st.button("Predict"):
    X_new = np.array([[credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
    X_new_scaled = scaler.transform(X_new)
    
    prediction = model.predict(X_new_scaled)
    st.write("⚠ Employee is likely to EXIT" if prediction[0][0] > 0.5 else "✅ Employee will STAY")


# In[ ]:




