import unittest
import numpy as np
from unittest.mock import MagicMock

# Mock model loading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_dummy_dnn_model():
    model = Sequential()
    model.add(Dense(10, input_dim=17, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Mock functions from your ASD tool
def calculate_aq_scores(responses):
    scores = []
    for i, response in enumerate(responses, 1):
        if i in [1, 7, 8, 10]:
            score = 1 if response in ["Definitely Agree", "Slightly Agree"] else 0
        else:
            score = 1 if response in ["Slightly Disagree", "Definitely Disagree"] else 0
        scores.append(score)
    return scores

def prepare_input_data(aq_scores, age, gender, ethnicity, jaundice, autism_family, country_hash):
    age_normalized = age / 120
    gender_encoded = 1 if gender == "Male" else 0
    jaundice_encoded = 1 if jaundice == "Yes" else 0
    autism_family_encoded = 1 if autism_family == "Yes" else 0
    result_score = sum(aq_scores) / 10
    return np.array(aq_scores + [age_normalized, gender_encoded, ethnicity, jaundice_encoded, autism_family_encoded, country_hash, result_score]).reshape(1, -1)

class TestASDScreening(unittest.TestCase):
    def setUp(self):
        self.dnn_model = create_dummy_dnn_model()
        self.dnn_model.predict = MagicMock(return_value=np.array([[1]]))  # Mock ASD prediction
        self.ml_model = MagicMock()
        self.ml_model.predict.return_value = [1]

    def test_aq_score_calculation(self):
        responses = [
            "Definitely Agree", "Definitely Disagree", "Slightly Disagree", "Definitely Disagree",
            "Definitely Disagree", "Slightly Disagree", "Definitely Agree", "Slightly Agree",
            "Definitely Disagree", "Slightly Agree"
        ]
        scores = calculate_aq_scores(responses)
        self.assertEqual(sum(scores), 10)

    def test_input_data_shape(self):
        scores = [1] * 10
        input_data = prepare_input_data(scores, 25, "Male", 0, "Yes", "Yes", 0.23)
        self.assertEqual(input_data.shape, (1, 17))

    def test_model_prediction(self):
        scores = [1] * 10
        input_data = prepare_input_data(scores, 25, "Male", 0, "Yes", "Yes", 0.23)
        pred_dnn = self.dnn_model.predict(input_data)[0][0]
        pred_ml = self.ml_model.predict(input_data)[0]
        self.assertEqual(int(round(pred_dnn)), 1)
        self.assertEqual(pred_ml, 1)

if __name__ == '__main__':
    unittest.main()
