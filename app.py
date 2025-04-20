import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
from info import ethnicity_mapping, ethnicity_code_mapping, aq_questions

st.set_page_config(page_title="ASD Screening Tool", layout="wide")

st.title("Autism Spectrum Disorder (ASD) Screening Tool")
st.markdown("""
This app uses both Machine Learning and Deep Learning models to screen for ASD based on the AQ-10 questionnaire.
""")

# Initialize database
def init_db():
    conn = sqlite3.connect('asd_screening.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS screenings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        country TEXT,
        ethnicity TEXT,
        jaundice TEXT,
        autism_family TEXT,
        aq_responses TEXT,
        aq_scores TEXT,
        total_score INTEGER,
        ml_prediction INTEGER,
        dnn_prediction INTEGER,
        timestamp DATETIME
    )
    ''')
    conn.commit()
    conn.close()

# Save data to database
def save_to_db(name, age, gender, country, ethnicity, jaundice, autism_family, 
               aq_responses, aq_scores, total_score, ml_prediction, dnn_prediction):
    conn = sqlite3.connect('asd_screening.db')
    c = conn.cursor()
    c.execute('''
    INSERT INTO screenings (
        name, age, gender, country, ethnicity, jaundice, autism_family,
        aq_responses, aq_scores, total_score, ml_prediction, dnn_prediction, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name, age, gender, country, ethnicity, jaundice, autism_family,
        str(aq_responses), str(aq_scores), total_score, ml_prediction, dnn_prediction, 
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

# Initialize the database when the app loads
init_db()

@st.cache_resource
def load_models():
    try:
        ml_model = pickle.load(open('ML_Model.pkl', 'rb'))
        dnn_model = load_model('DNN_Model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    return ml_model, dnn_model

ml_model, dnn_model = load_models()

# Creating two tabs for Form and Results
tab1, tab2 = st.tabs(["📝 Screening Form", "📊 Results"])

with tab1:
    with st.form("asd_form"):
        # Add name field at the top
        name = st.text_input("Full Name", placeholder="Enter your name")
        
        aq_responses = []
        for i, question in enumerate(aq_questions, 1):
            st.write(f"**Q{i}: {question}**")
            response = st.radio(
                "", 
                options=("Definitely Agree", "Slightly Agree", "Slightly Disagree", "Definitely Disagree"),
                key=f"q{i}",
                horizontal=True,
                index=None
            )
            aq_responses.append(response)

        st.subheader("Additional Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age")
            gender = st.selectbox("Gender", ["", "Male", "Female"], index=0)
            country = st.selectbox("Country", [""] + sorted(list(ethnicity_mapping.keys())))
            ethnicity = ethnicity_mapping.get(country, "Others")
            if country:
                st.info(f"Ethnicity: {ethnicity}")

        with col2:
            jaundice = st.selectbox("History of Jaundice at Birth", ["", "Yes", "No"], index=0)
            autism_family = st.selectbox("Family Member with Autism", ["", "Yes", "No"], index=0)

        submitted = st.form_submit_button("Submit for Screening")

with tab2:
    if submitted:
        if (None in aq_responses or "" in aq_responses or not age or gender == "" or 
            country == "" or jaundice == "" or autism_family == "" or not name):
            st.error("Please fill all the fields in the form tab before viewing results.")
        else:
            aq_scores = []
            for i, response in enumerate(aq_responses, 1):
                if i in [1, 7, 8, 10]:
                    score = 1 if response in ["Definitely Agree", "Slightly Agree"] else 0
                else:
                    score = 1 if response in ["Slightly Disagree", "Definitely Disagree"] else 0
                aq_scores.append(score)

            gender_encoded = 1 if gender == "Male" else 0
            ethnicity_encoded = ethnicity_code_mapping.get(ethnicity, 9)
            age_normalized = age / 120
            jaundice_encoded = 1 if jaundice == "Yes" else 0
            autism_family_encoded = 1 if autism_family == "Yes" else 0
            country_encoded = hash(country) % 100 / 100
            result_score = sum(aq_scores)

            input_data = np.array(
                aq_scores + [
                    age_normalized, gender_encoded, ethnicity_encoded, 
                    jaundice_encoded, autism_family_encoded, country_encoded, result_score
                ]
            ).reshape(1, -1)

            try:
                prediction_ml = ml_model.predict(input_data)[0]
                prediction_dnn_prob = dnn_model.predict(input_data, verbose=0)[0][0]
                prediction_dnn = int(prediction_dnn_prob >= 0.5)
                
                # Save data to the database after successful predictions
                save_to_db(
                    name, age, gender, country, ethnicity, jaundice, autism_family,
                    aq_responses, aq_scores, sum(aq_scores), int(prediction_ml), prediction_dnn
                )
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.subheader("Screening Results")

            # Create a More Structured Layout for Results
            st.markdown(f"### Results for: {name}")
            st.markdown("### 🔍 **Model-Based Results**")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💡 **Machine Learning Model**")
                if prediction_ml == 1:
                    st.error("**Possible ASD traits detected**. It is recommended to seek a professional assessment.")
                else:
                    st.success("**No significant ASD traits detected** based on the Machine Learning model.")

            with col2:
                st.markdown("#### 🤖 **Deep Learning Model**")
                if prediction_dnn == 1:
                    st.error("**Possible ASD traits detected**. It is recommended to seek a professional assessment.")
                else:
                    st.success("**No significant ASD traits detected** based on the Deep Learning model.")

            st.markdown("### 🧮 **Your AQ-10 Score**")
            st.info(f"**Score: {sum(aq_scores)}/10**")

            st.markdown("### 📋 **Your Inputs Summary**")
            st.write("#### **Demographics**")
            st.table(pd.DataFrame({
                "Category": ["Name", "Age", "Gender", "Country", "Ethnicity", "Jaundice", "Family Member with ASD"],
                "Response": [name, age, gender, country, ethnicity, jaundice, autism_family]
            }))

            st.write("#### **AQ-10 Responses Breakdown**")
            st.table(pd.DataFrame({
                "Question #": [f"Q{i}" for i in range(1, 11)],
                "Question": aq_questions,
                "Response": aq_responses,
                "Score": aq_scores
            }))

            st.write("#### **Model Input Features**")
            feature_names = [f"A{i}_Score" for i in range(1, 11)] + [
                "Age (normalized)", "Gender", "Ethnicity", 
                "Jaundice", "Family_ASD", "Country", "AQ_Result_Score"
            ]
            st.table(pd.DataFrame({
                "Feature": feature_names,
                "Value": [f"{v:.4f}" if isinstance(v, float) else v for v in input_data.flatten()]
            }))

            st.markdown("### 📝 **Conclusion**")
            if prediction_ml == 1 or prediction_dnn == 1:
                st.warning("""
                    Based on the models' predictions and your AQ-10 score, there may be some traits consistent with ASD.
                    **However, please note**: This is not a diagnostic tool, and you should seek a professional assessment for further evaluation.
                """)
            else:
                st.success("""
                    Based on the results, there is no indication of significant ASD traits.
                    **However**, if you have concerns or observe other symptoms, consider consulting a professional for further evaluation.
                """)
            
            st.success("Your screening data has been saved.")
            st.warning("""**Disclaimer**: This is not a diagnostic tool. Please consult a licensed medical professional for diagnosis.""")

    else:
        st.info("Submit the form in the 'Screening Form' tab to view results here.")

st.sidebar.header("References")
st.sidebar.markdown("""
- [Learn more about ASD diagnosis](https://www.cdc.gov/ncbddd/autism/)
""")