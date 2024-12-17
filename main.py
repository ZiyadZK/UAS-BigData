import numpy as np
from joblib import load
from flask import Flask, request, jsonify
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "UAS BIG DATA"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Allowed model names
        allowed_model_name = [
            'gaussiannb_model',
            'knn_model',
            'logistic_regression_model',
            'svm_model'
        ]

        # Retrieve model name from query or use default
        model_name = request.args.get('model', 'logistic_regression_model')
        if model_name not in allowed_model_name:
            model_name = 'logistic_regression_model'

        # Load the model
        model = load(model_name + '.joblib')

        # Get JSON data from the request
        input_data = request.get_json()

        # Preprocess input data
        features = np.array([
            int(input_data['Sex']),
            int(input_data['Age']),
            int(input_data['Passenger Class']),
            int(input_data['No of Parents or Children on Board']),
            int(input_data['No of Siblings or Spouses on Board']),
            float(input_data['Passenger Fare']),
            int(input_data['Port of Embarkation'])
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)  # Get class probabilities

        # Convert probabilities to floats out of 100
        prob_not_survived = round(probabilities[0][0] * 100, 1)
        prob_survived = round(probabilities[0][1] * 100, 1)

        # Response JSON
        return jsonify({
            'model_name': model_name,
            'survived': True if int(prediction[0]) == 1 else False,
            'probabilities': {
                'not_survived': prob_not_survived,
                'survived': prob_survived
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
