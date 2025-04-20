import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="ASD Screening Tool", layout="wide")

st.title("Autism Spectrum Disorder (ASD) Screening Tool")
st.markdown("""
This app uses both Machine Learning and Deep Learning models to screen for ASD based on the AQ-10 questionnaire.
""")

# Country to ethnicity mapping
ethnicity_mapping = {
    'United States': 'White European',
    'Brazil': 'South American',
    'Spain': 'White European',
    'Egypt': 'Middle Eastern',
    'New Zealand': 'Pasifika',
    'Bahamas': 'Black',
    'Burundi': 'African',
    'Austria': 'White European',
    'Argentina': 'South American',
    'Jordan': 'Middle Eastern',
    'Ireland': 'White European',
    'United Arab Emirates': 'Middle Eastern',
    'Afghanistan': 'South Asian',
    'Lebanon': 'Middle Eastern',
    'United Kingdom': 'White European',
    'South Africa': 'Black',
    'Italy': 'White European',
    'Pakistan': 'South Asian',
    'Bangladesh': 'South Asian',
    'Chile': 'South American',
    'France': 'White European',
    'China': 'Asian',
    'Australia': 'White European',
    'Canada': 'White European',
    'Saudi Arabia': 'Middle Eastern',
    'Netherlands': 'White European',
    'Romania': 'White European',
    'Sweden': 'White European',
    'Tonga': 'Pasifika',
    'Oman': 'Middle Eastern',
    'India': 'South Asian',
    'Philippines': 'Asian',
    'Sri Lanka': 'South Asian',
    'Sierra Leone': 'African',
    'Ethiopia': 'African',
    'Viet Nam': 'Asian',
    'Iran': 'Middle Eastern',
    'Costa Rica': 'Latino',
    'Germany': 'White European',
    'Mexico': 'Latino',
    'Russia': 'White European',
    'Armenia': 'White European',
    'Iceland': 'White European',
    'Nicaragua': 'Latino',
    'Hong Kong': 'Asian',
    'Japan': 'Asian',
    'Ukraine': 'White European',
    'Kazakhstan': 'Asian',
    'American Samoa': 'Pasifika',
    'Uruguay': 'South American',
    'Serbia': 'White European',
    'Portugal': 'White European',
    'Malaysia': 'Asian',
    'Ecuador': 'Latino',
    'Niger': 'African',
    'Belgium': 'White European',
    'Bolivia': 'South American',
    'Aruba': 'Latino',
    'Finland': 'White European',
    'Turkey': 'White European',
    'Nepal': 'South Asian',
    'Indonesia': 'Asian',
    'Angola': 'African',
    'Azerbaijan': 'Middle Eastern',
    'Iraq': 'Middle Eastern',
    'Czech Republic': 'White European',
    'Cyprus': 'Middle Eastern'
}

ethnicity_code_mapping = {
    'White European': 0,
    'South American': 1,
    'Asian': 2,
    'Black': 3,
    'Middle Eastern': 4,
    'South Asian': 5,
    'African': 6,
    'Latino': 7,
    'Pasifika': 8,
    'Others': 9
}

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

aq_questions = [
    "I often notice small sounds when others do not",
    "I usually concentrate more on the whole picture, rather than the small details",
    "I find it easy to do more than one thing at once",
    "If there is an interruption, I can switch back to what I was doing very quickly",
    "I find it easy to 'read between the lines' when someone is talking to me",
    "I know how to tell if someone listening to me is getting bored",
    "When I'm reading a story, I find it difficult to work out the characters' intentions",
    "I like to collect information about categories of things",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face",
    "I find it difficult to work out people's intentions"
]

with st.form("asd_form"):
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

if submitted:
    if (None in aq_responses or "" in aq_responses or not age or gender == "" or 
        country == "" or jaundice == "" or autism_family == ""):
        st.error("Please fill all the fields.")
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
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.subheader("Screening Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💡 Machine Learning Model")
            if prediction_ml == 1:
                st.error("Possible ASD traits detected.")
            else:
                st.success("No significant ASD traits detected.")

        with col2:
            st.markdown("### 🤖 Deep Learning Model")
            if prediction_dnn == 1:
                st.error("Possible ASD traits detected.")
            else:
                st.success("No significant ASD traits detected.")

        st.markdown("### 🧮 Your AQ-10 Score")
        st.info(f"**Score: {sum(aq_scores)}/10**")

        st.markdown("### 📋 Your Inputs")
        st.write("**Demographics:**")
        st.table(pd.DataFrame({
            "Category": ["Age", "Gender", "Country", "Ethnicity", "Jaundice", "Family ASD"],
            "Response": [age, gender, country, ethnicity, jaundice, autism_family]
        }))

        st.write("**AQ-10 Responses:**")
        st.table(pd.DataFrame({
            "Question #": [f"Q{i}" for i in range(1, 11)],
            "Question": aq_questions,
            "Response": aq_responses,
            "Score": aq_scores
        }))

        st.write("**Model Input Features:**")
        feature_names = [f"A{i}_Score" for i in range(1, 11)] + [
            "Age (normalized)", "Gender", "Ethnicity", 
            "Jaundice", "Family_ASD", "Country", "AQ_Result_Score"
        ]
        st.table(pd.DataFrame({
            "Feature": feature_names,
            "Value": [f"{v:.4f}" if isinstance(v, float) else v for v in input_data.flatten()]
        }))

        st.warning("""
        **Disclaimer**: This is not a diagnostic tool. Please consult a licensed medical professional for diagnosis.
        """)

st.sidebar.header("References")
st.sidebar.markdown("""
- [Learn more about ASD diagnosis](https://www.cdc.gov/ncbddd/autism/)
""")
