
import os 
import pickle 
import streamlit as st 
from streamlit_option_menu import option_menu 

# Set page configuration 
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="💉")

# Load models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb')) 
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb')) 

# Sidebar
with st.sidebar: 
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'], 
        menu_icon='hospital-fill', 
        icons=['activity', 'heart'], 
        default_index=0
    )

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
            st.warning("⚠️ Untreated diabetes may lead to:\n"
                       "- Kidney Disease\n"
                       "- Heart Disease\n"
                       "- Vision Loss")
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('Thal (0=normal; 1=fixed; 2=reversible)')

    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])
        
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person has heart disease'
            st.warning("⚠️ Untreated heart disease may lead to:\n"
                       "- Stroke\n"
                       "- Kidney Damage\n"
                       "- Heart Failure")
        else:
            heart_diagnosis = 'The person does not have heart disease'

    st.success(heart_diagnosis)
