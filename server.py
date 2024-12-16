import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import joblib
import logging

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Load model and data
try:
    model = pickle.load(open('RandomForest.pkl', 'rb'))
    symptoms_df = pd.read_csv('symptoms_df.csv')
    description = pd.read_csv('description.csv')
    precautions = pd.read_csv('precautions_df.csv')
    medications = pd.read_csv('medications.csv')
    diets = pd.read_csv('diets.csv')
    workout = pd.read_csv('workout_df.csv')
    logging.info("All files loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    raise
except Exception as e:
    logging.error(f"Error loading files: {e}")
    raise

# Preprocess data for efficiency
symptom_list = symptoms_df.columns[:-1].tolist()

def correct_symptoms(input_symptoms):
    """Correct misspelled symptoms using fuzzy matching."""
    corrected = []
    for symptom in input_symptoms:
        best_match, score = process.extractOne(symptom, symptom_list)
        if score >= 80:  # Adjust threshold as needed
            corrected.append(best_match)
        else:
            logging.warning(f"No good match for symptom: {symptom}")
    return corrected

def information(predicted_dis):
    """Retrieve detailed information about the predicted disease."""
    return {
        "description": " ".join(description[description['Disease'] == predicted_dis]['Description']),
        "precautions": precautions[precautions['Disease'] == predicted_dis].iloc[:, 1:].values.tolist(),
        "medications": medications[medications['Disease'] == predicted_dis]['Medication'].values,
        "diet": diets[diets['Disease'] == predicted_dis]['Diet'].values,
        "workout": workout[workout['disease'] == predicted_dis]['workout'].values
    }



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input symptoms
        input_data = request.form['symptoms']
        symptoms = [s.strip() for s in input_data.split(',')]
        logging.info(f"Input symptoms: {symptoms}")

        # Correct symptoms
        corrected_symptoms = correct_symptoms(symptoms)
        logging.info(f"Corrected symptoms: {corrected_symptoms}")

        # Create input vector
        input_vector = np.zeros(len(symptom_list))
        for symptom in corrected_symptoms:
            if symptom in symptom_list:
                input_vector[symptom_list.index(symptom)] = 1

        # Predict disease
        predicted_disease = model.predict([input_vector])[0]
        logging.info(f"Predicted disease: {predicted_disease}")

        # Get additional information
        disease_info = information(predicted_disease)

        # Render or return JSON
        if request.is_json:
            return jsonify({
                "symptoms": corrected_symptoms,
                "predicted_disease": predicted_disease,
                "information": disease_info
            })
        return render_template(
            'result.html',
            symptoms=corrected_symptoms,
            disease=predicted_disease,
            info=disease_info
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500

if __name__ == '__main__':
    app.run(debug=True)
