import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained ANN model
# model = tf.keras.models.load_model('model\model.pkl')

# Load the model
model = load_model(r'C:\Personal\Programming & Crypto\Jigsaw\alz\model\ann.h5')

# Set up the app title
st.title("Alzheimer's Disease Prediction")


# Demographic Details
st.header('Demographic Details')
age = st.slider('Age', 60, 90, 65)
gender = st.selectbox('Gender', ['Male', 'Female'])
ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'])
education = st.selectbox('Education Level', ['None', 'High School', 'Bachelor\'s', 'Higher'])

# Lifestyle Factors
st.header('Lifestyle Factors')
bmi = st.slider('Body Mass Index (BMI)', 15.0, 40.0, 25.0)
smoking = st.selectbox('Smoking Status', ['No', 'Yes'])
alcohol = st.slider('Weekly Alcohol Consumption (Units)', 0, 20, 5)
physical_activity = st.slider('Weekly Physical Activity (Hours)', 0, 10, 5)
diet_quality = st.slider('Diet Quality Score', 0, 10, 7)
sleep_quality = st.slider('Sleep Quality Score', 4, 10, 7)

# Medical History
st.header('Medical History')
family_history = st.selectbox('Family History of Alzheimer’s Disease', ['No', 'Yes'])
cardiovascular_disease = st.selectbox('Cardiovascular Disease', ['No', 'Yes'])
diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
depression = st.selectbox('Depression', ['No', 'Yes'])
head_injury = st.selectbox('Head Injury', ['No', 'Yes'])
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])

# Clinical Measurements
st.header('Clinical Measurements')
systolic_bp = st.slider('Systolic Blood Pressure (mmHg)', 90, 180, 120)
diastolic_bp = st.slider('Diastolic Blood Pressure (mmHg)', 60, 120, 80)
chol_total = st.slider('Total Cholesterol (mg/dL)', 150, 300, 200)
chol_ldl = st.slider('LDL Cholesterol (mg/dL)', 50, 200, 100)
chol_hdl = st.slider('HDL Cholesterol (mg/dL)', 20, 100, 50)
chol_trig = st.slider('Triglycerides (mg/dL)', 50, 400, 150)

# Cognitive and Functional Assessments
st.header('Cognitive and Functional Assessments')
mmse = st.slider('MMSE Score', 0, 30, 25)
functional_assessment = st.slider('Functional Assessment Score', 0, 10, 7)
memory_complaints = st.selectbox('Memory Complaints', ['No', 'Yes'])
behavioral_problems = st.selectbox('Behavioral Problems', ['No', 'Yes'])
adl = st.slider('ADL Score', 0, 10, 8)

# Symptoms
st.header('Symptoms')
confusion = st.selectbox('Confusion', ['No', 'Yes'])
disorientation = st.selectbox('Disorientation', ['No', 'Yes'])
personality_changes = st.selectbox('Personality Changes', ['No', 'Yes'])
difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', ['No', 'Yes'])
forgetfulness = st.selectbox('Forgetfulness', ['No', 'Yes'])

# Prepare the input data for prediction
def prepare_data():
    gender_val = 0 if gender == 'Male' else 1
    ethnicity_val = ['Caucasian', 'African American', 'Asian', 'Other'].index(ethnicity)
    education_val = ['None', 'High School', 'Bachelor\'s', 'Higher'].index(education)
    smoking_val = 1 if smoking == 'Yes' else 0
    family_history_val = 1 if family_history == 'Yes' else 0
    cardiovascular_val = 1 if cardiovascular_disease == 'Yes' else 0
    diabetes_val = 1 if diabetes == 'Yes' else 0
    depression_val = 1 if depression == 'Yes' else 0
    head_injury_val = 1 if head_injury == 'Yes' else 0
    hypertension_val = 1 if hypertension == 'Yes' else 0
    memory_complaints_val = 1 if memory_complaints == 'Yes' else 0
    behavioral_val = 1 if behavioral_problems == 'Yes' else 0
    confusion_val = 1 if confusion == 'Yes' else 0
    disorientation_val = 1 if disorientation == 'Yes' else 0
    personality_val = 1 if personality_changes == 'Yes' else 0
    difficulty_val = 1 if difficulty_completing_tasks == 'Yes' else 0
    forgetfulness_val = 1 if forgetfulness == 'Yes' else 0

    return np.array([age, gender_val, ethnicity_val, education_val, bmi, smoking_val, alcohol,
                     physical_activity, diet_quality, sleep_quality, family_history_val, cardiovascular_val,
                     diabetes_val, depression_val, head_injury_val, hypertension_val, systolic_bp, diastolic_bp,
                     chol_total, chol_ldl, chol_hdl, chol_trig, mmse, functional_assessment, memory_complaints_val,
                     behavioral_val, adl, confusion_val, disorientation_val, personality_val, difficulty_val,
                     forgetfulness_val]).reshape(1, -1)

# Button to make a prediction
if st.button('Predict Alzheimer\'s Diagnosis'):
    input_data = prepare_data()
    prediction = model.predict(input_data)
    diagnosis = 'Yes' if prediction[0] > 0.5 else 'No'
    st.write(f'Predicted Diagnosis: Alzheimer\'s Disease - {diagnosis}')
