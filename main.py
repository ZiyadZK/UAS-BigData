import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from flask import Flask, request, jsonify
import os

# Step 1: Data Preprocessing

def preprocess_training_data(df):
    # Feature engineering
    df['Sex'] = df['Sex'].map({'Female': 1, 'Male': 0})
    df['Passenger Class'] = df['Passenger Class'].map({'First': 1, 'Second': 2, 'Third': 3})
    df['Port of Embarkation'] = df['Port of Embarkation'].fillna(df['Port of Embarkation'].mode()[0])


    port_mapping = {'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3}
    df['Port of Embarkation'] = df['Port of Embarkation'].map(port_mapping)

    df['Port of Embarkation'] = df['Port of Embarkation'].fillna(0)

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Port of Embarkation'] = df['Port of Embarkation'].fillna(0)

    # Selecting features and target
    features = df[['Sex', 'Age', 'Passenger Class', 'No of Siblings or Spouses on Board', 'No of Parents or Children on Board', 'Port of Embarkation']]
    target = df['Survived']
    return features, target

# Load dataset (replace 'titanic.csv' with your dataset path)
df = pd.read_csv('titanic.csv')
X, y = preprocess_training_data(df)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Model Training

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 3: API Deployment

app = Flask(__name__)

# Load pre-trained model and scaler
model = load('logistic_regression_model.joblib')

def preprocess_input(data):
    # Map categorical variables
    data['Sex'] = {'Female': 1, 'Male': 0}.get(data['Sex'], -1)
    data['Passenger Class'] = {'First': 1, 'Second': 2, 'Third': 3}.get(data['Passenger Class'], -1)
    data['Port of Embarkation'] = {'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3}.get(data.get('Port of Embarkation', ''), 0)

    # Handle missing values
    data['Age'] = data.get('Age', 30)
    data['No of Parents or Children on Board'] = data.get('No of Parents or Children on Board', 0)
    data['No of Siblings or Spouses on Board'] = data.get('No of Siblings or Spouses on Board', 0)

    # Prepare the feature array
    features = np.array([
        data['Sex'],
        data['Age'],
        data['Passenger Class'],
        data['No of Parents or Children on Board'],
        data['No of Siblings or Spouses on Board'],
        data['Port of Embarkation']
    ]).reshape(1, -1)

    # Standardize the features
    return scaler.transform(features)

@app.route('/')
def home():
    return "UAS BIG DATA"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Preprocess input data
        features = preprocess_input(input_data)

        print(features)

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None

        # Map the prediction to "Survived" or "Not Survived"
        prediction_label = "Survived" if prediction[0] == 1 else "Not Survived"
        probability_label = {
            "Survived": probability[1] if probability is not None else None,
            "Not Survived": probability[0] if probability is not None else None
        } if probability is not None else None

        return jsonify({
            "prediction": prediction_label,
            "probability": probability_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))