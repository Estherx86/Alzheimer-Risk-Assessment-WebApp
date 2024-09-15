import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained ANN model
# model = tf.keras.models.load_model('model\model.pkl')

# Load the model
model = load_model(r'C:\Personal\Programming & Crypto\Jigsaw\alz\model\ann.h5')

# Set up the app title
st.title("Alzheimer's Disease Prediction")

# # Create input fields for each variable
# age = st.number_input('Age', min_value=60, max_value=100, step=1)
# gender = st.selectbox('Gender', [0, 1])  # Assuming 0 = Male, 1 = Female
# ethnicity = st.selectbox('Ethnicity', [0, 1, 2, 3])  # Add your actual categories here
# education_level = st.selectbox('Education Level', [0, 1, 2, 3, 4])  # Add your actual categories here
# bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, step=0.1)
# smoking = st.selectbox('Smoking', [0, 1])
# alcohol_consumption = st.number_input('Alcohol Consumption', min_value=0.0, max_value=50.0, step=0.1)
# physical_activity = st.number_input('Physical Activity (hours)', min_value=0.0, max_value=10.0, step=0.1)
# diet_quality = st.number_input('Diet Quality (score)', min_value=0.0, max_value=10.0, step=0.1)
# sleep_quality = st.number_input('Sleep Quality (score)', min_value=0.0, max_value=10.0, step=0.1)
# family_history = st.selectbox('Family History of Alzheimer\'s', [0, 1])
# cardiovascular_disease = st.selectbox('Cardiovascular Disease', [0, 1])
# diabetes = st.selectbox('Diabetes', [0, 1])
# depression = st.selectbox('Depression', [0, 1])
# head_injury = st.selectbox('Head Injury', [0, 1])
# hypertension = st.selectbox('Hypertension', [0, 1])
# systolic_bp = st.number_input('Systolic Blood Pressure', min_value=90, max_value=200, step=1)
# diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=60, max_value=120, step=1)
# cholesterol_total = st.number_input('Total Cholesterol', min_value=100, max_value=300, step=1)
# cholesterol_ldl = st.number_input('Cholesterol LDL', min_value=50, max_value=200, step=1)
# cholesterol_hdl = st.number_input('Cholesterol HDL', min_value=30, max_value=100, step=1)
# cholesterol_triglycerides = st.number_input('Cholesterol Triglycerides', min_value=50, max_value=300, step=1)
# mmse = st.number_input('MMSE Score', min_value=0.0, max_value=30.0, step=0.1)
# functional_assessment = st.number_input('Functional Assessment (score)', min_value=0.0, max_value=10.0, step=0.1)
# memory_complaints = st.selectbox('Memory Complaints', [0, 1])
# behavioral_problems = st.selectbox('Behavioral Problems', [0, 1])
# adl = st.number_input('ADL (score)', min_value=0.0, max_value=10.0, step=0.1)
# confusion = st.selectbox('Confusion', [0, 1])
# disorientation = st.selectbox('Disorientation', [0, 1])
# personality_changes = st.selectbox('Personality Changes', [0, 1])
# difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', [0, 1])
# forgetfulness = st.selectbox('Forgetfulness', [0, 1])

# # Button to make prediction
# if st.button('Predict Diagnosis'):
#     # Convert the inputs to a numpy array
#     user_input = np.array([[age, gender, ethnicity, education_level, bmi, smoking,
#                             alcohol_consumption, physical_activity, diet_quality, sleep_quality,
#                             family_history, cardiovascular_disease, diabetes, depression, head_injury,
#                             hypertension, systolic_bp, diastolic_bp, cholesterol_total, cholesterol_ldl,
#                             cholesterol_hdl, cholesterol_triglycerides, mmse, functional_assessment,
#                             memory_complaints, behavioral_problems, adl, confusion, disorientation,
#                             personality_changes, difficulty_completing_tasks, forgetfulness]])
    
#     # Make prediction
#     prediction = model.predict(user_input)
    
#     # Show result
#     st.write(f"The predicted diagnosis is: {'Alzheimer' if prediction[0] > 0.5 else 'No Alzheimer'}")


# import streamlit as st
# import numpy as np
# from keras.models import load_model

# # Load the model (ensure the model is available in the path)
# model = load_model('path_to_model/ann.h5')

# # Title of the web app
# st.title('Alzheimer’s Disease Prediction App')

# # Patient Information
# st.header('Patient Information')
# patient_id = st.text_input('Patient ID', '4751')

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
