# Alzheimer's Risk Assessment Web Application

This repository contains a **Streamlit-based web application** designed to assess the risk of Alzheimer's disease using patient information. The model is powered by a trained Artificial Neural Network (ANN) to predict Alzheimer's risk based on clinical and demographic features provided by the user.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Usage](#usage)

---

## Overview

Alzheimer's disease is a progressive neurological disorder that affects memory and cognitive abilities. Early detection of the risk can help with timely medical interventions and improved patient outcomes.

This web application allows healthcare practitioners and individuals to:
- Input patient information such as age, gender, education level, and lifestyle factors.
- Predict the likelihood of Alzheimer's disease based on the provided data.
- View results directly on the user-friendly web interface.

---

## Features

- **Interactive Form:** Collects user input for clinical and demographic factors.
- **Risk Assessment Model:** Predicts Alzheimer's risk using a trained ANN.
- **User-Friendly Interface:** Developed with Streamlit for intuitive navigation.
- **Real-Time Results:** Delivers predictions instantly based on user input.

---

## Dataset

The model was trained on a dataset containing diverse clinical and demographic attributes, such as:
- Age
- Gender
- Ethnicity
- Education Level
- Lifestyle factors (e.g., Smoking, BMI, etc.)

The dataset includes 34 features, encompassing both numerical and categorical data, with the target variable being `Diagnosis` (Alzheimer's or not).

---


## Technology Stack

- **Frontend:** Streamlit
- **Backend:** Python (with XGBClassifier)
- **Data Handling:** Pandas, NumPy
- **Deployment:** Streamlit web app
- **Visualization:** Matplotlib, Seaborn

---

## Usage

### Clone the Repository
```bash
git clone https://github.com/mohammadreza-mohammadi94/Alzheimer-Risk-Assessment-WebApp.git
cd Alzheimer-Risk-Assessment-WebApp
```

### Install Dependencies
Install the required libraries using

### Run the Application
Launch the web application locally:
```bash
streamlit run app.py
```

### Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

### Input Patient Data
Fill out the form with the required patient information and submit to view the prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check out the [issues page](https://github.com/mohammadreza-mohammadi94/Alzheimer-Risk-Assessment-WebApp/issues).

---

This project aims to assist in early detection of Alzheimer's risk and is a step forward in leveraging AI for healthcare applications.
