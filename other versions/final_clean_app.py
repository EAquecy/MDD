
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="💉")

# Load models
try:
    diabetes_model = pickle.load(open('../diabetes.sav', 'rb'))
    heart_disease_model = pickle.load(open('../heart.sav', 'rb'))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart'],
        default_index=0
    )

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies')
    with col2: Glucose = st.text_input('Glucose Level')
    with col3: BloodPressure = st.text_input('Blood Pressure value')
    with col1: SkinThickness = st.text_input('Skin Thickness value')
    with col2: Insulin = st.text_input('Insulin Level')
    with col3: BMI = st.text_input('BMI value')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2: Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        try:
            user_input = np.array([[float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                                    float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]])
            prediction = diabetes_model.predict(user_input)
            if prediction[0] == 1:
                st.error('The person is diabetic')
                st.warning("⚠️ Untreated diabetes may lead to:\n"
                           "- Kidney Disease\n"
                           "- Heart Disease\n"
                           "- Vision Loss")
            else:
                st.success('The person is not diabetic')
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age')
    with col2: sex = st.text_input('Sex')
    with col3: cp = st.text_input('Chest Pain types')
    with col1: trestbps = st.text_input('Resting Blood Pressure')
    with col2: chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1: restecg = st.text_input('Resting Electrocardiographic results')
    with col2: thalach = st.text_input('Maximum Heart Rate achieved')
    with col3: exang = st.text_input('Exercise Induced Angina')
    with col1: oldpeak = st.text_input('ST depression induced by exercise')
    with col2: slope = st.text_input('Slope of the peak exercise ST segment')
    with col3: ca = st.text_input('Major vessels colored by flourosopy')
    with col1: thal = st.text_input('Thal (0=normal; 1=fixed; 2=reversible)')

    if st.button('Heart Disease Test Result'):
        try:
            user_input = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol),
                                    float(fbs), float(restecg), float(thalach), float(exang),
                                    float(oldpeak), float(slope), float(ca), float(thal)]])
            prediction = heart_disease_model.predict(user_input)
            if prediction[0] == 1:
                st.error('The person has heart disease')
                st.warning("⚠️ Untreated heart disease may lead to:\n"
                           "- Stroke\n"
                           "- Kidney Damage\n"
                           "- Heart Failure")
            else:
                st.success('The person does not have heart disease')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
