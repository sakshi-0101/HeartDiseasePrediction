import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


def binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href


st.title("Heart Disease Predictor")

tab1, tab2 = st.tabs(['Predict', 'Bulk Predict'])

# ===================== TAB 1 =====================
with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    # Encoding
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algonames = ['Decision Tree', 'Logistic Regression', 'Support Vector Machine']
    modelnames = ['tree.pkl', 'LogisticR.pkl', 'SVM.pkl']

    def predict_heart_disease(data):
        results = []
        for modelname in modelnames:
            with open(modelname, 'rb') as f:
                model = pickle.load(f)
            pred = model.predict(data)
            results.append(pred)
        return results

    if st.button("Submit"):
        st.subheader("Results")
        st.markdown('----------------')

        result = predict_heart_disease(input_data)

        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart Disease detected.")
            st.markdown('------------------')


# ===================== TAB 2 =====================
with tab2:
    st.title('Upload CSV File')
    st.subheader('Instructions before uploading:')

    st.info("""
    1. No NaN values allowed.
    2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').
    3. Check the spellings of the feature names.
    4. Feature values conventions:

        - Age: age of the patient [years]
        - Sex: sex of the patient [0: Male, 1: Female]
        - ChestPainType: chest pain type 
            [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]
        - RestingBP: resting blood pressure [mm Hg]
        - Cholesterol: serum cholesterol [mm/dl]
        - FastingBS: fasting blood sugar 
            [1: if FastingBS > 120 mg/dl, 0: otherwise]
        - RestingECG: resting electrocardiogram results 
            [0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy]
        - MaxHR: maximum heart rate achieved 
            [Numeric value between 60 and 202]
        - ExerciseAngina: exercise-induced angina 
            [1: Yes, 0: No]
        - Oldpeak: ST depression induced by exercise
        - ST_Slope: slope of the peak exercise ST segment 
            [0: upsloping, 1: flat, 2: downsloping]
""")

    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        expected_columns = [
            'Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
            'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'
        ]

        if all(col in input_data.columns for col in expected_columns):

            # correct order
            input_data = input_data[expected_columns]

            model = pickle.load(open('LogisticR.pkl', 'rb'))

            predictions = model.predict(input_data)

            input_data['Prediction'] = predictions

            st.subheader("Predictions:")
            st.write(input_data)

            st.markdown(binary_file_downloader_html(input_data), unsafe_allow_html=True)

        else:
            st.warning("CSV must have correct column names.")

    else:
        st.info("Upload a CSV file to get predictions.")



   