import unittest
import os
import sqlite3
import numpy as np
import pickle
from tensorflow.keras.models import load_model

DB_PATH = "asd_screening.db"

# Sample dummy AQ responses and demographics
DUMMY_AQ_SCORES = [1] * 10
AGE = 25
GENDER = "Male"
ETHNICITY = "Others"
JAUNDICE = "Yes"
FAMILY_ASD = "Yes"
COUNTRY = "Testland"
COUNTRY_HASH = hash(COUNTRY) % 100 / 100
RESULT_SCORE = sum(DUMMY_AQ_SCORES)

# --- Load models (as done in app.py) ---
def load_ml_model(path='ML_Model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_dnn_model(path='DNN_Model.h5'):
    return load_model(path)

# --- Prepare input data ---
def prepare_input(aq_scores):
    age_normalized = AGE / 120
    gender_encoded = 1 if GENDER == "Male" else 0
    jaundice_encoded = 1 if JAUNDICE == "Yes" else 0
    autism_family_encoded = 1 if FAMILY_ASD == "Yes" else 0
    return np.array(
        aq_scores + [age_normalized, gender_encoded, 9,  # Ethnicity code dummy
                     jaundice_encoded, autism_family_encoded, COUNTRY_HASH, RESULT_SCORE]
    ).reshape(1, -1)

# --- DB check ---
def get_last_entry():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, total_score, ml_prediction, dnn_prediction FROM screenings ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row

# --- Test Class ---
class TestAppIntegration(unittest.TestCase):

    def setUp(self):
        self.ml_model = load_ml_model()
        self.dnn_model = load_dnn_model()

    def test_models_load_successfully(self):
        self.assertIsNotNone(self.ml_model)
        self.assertIsNotNone(self.dnn_model)

    def test_model_prediction_flow(self):
        input_data = prepare_input(DUMMY_AQ_SCORES)

        pred_ml = self.ml_model.predict(input_data)[0]
        pred_dnn_prob = self.dnn_model.predict(input_data, verbose=0)[0][0]
        pred_dnn = int(pred_dnn_prob >= 0.5)

        self.assertIn(pred_ml, [0, 1])
        self.assertIn(pred_dnn, [0, 1])

    def test_database_insertion(self):
        from app import save_to_db  # Import actual function

        save_to_db(
            name="IntegrationTestUser",
            age=AGE,
            gender=GENDER,
            country=COUNTRY,
            ethnicity=ETHNICITY,
            jaundice=JAUNDICE,
            autism_family=FAMILY_ASD,
            aq_responses=["Test"] * 10,
            aq_scores=DUMMY_AQ_SCORES,
            total_score=RESULT_SCORE,
            ml_prediction=1,
            dnn_prediction=1
        )

        entry = get_last_entry()
        self.assertIsNotNone(entry)
        self.assertEqual(entry[0], "IntegrationTestUser")
        self.assertEqual(entry[1], RESULT_SCORE)

if __name__ == '__main__':
    unittest.main()
