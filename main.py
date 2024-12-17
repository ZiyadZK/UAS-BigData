import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import os
from flask_cors import CORS
from joblib import load

app = Flask(__name__)
CORS(app)

DATASET_PATH = "titanic.csv"
if os.path.exists(DATASET_PATH):
    titanic_df = pd.read_csv(DATASET_PATH)
else:
    raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' not found!")

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
        features = np.array([[
            int(input_data['Sex']),
            int(input_data['Age']),
            int(input_data['Passenger Class']),
            int(input_data['No of Parents or Children on Board']),
            int(input_data['No of Siblings or Spouses on Board']),
            float(input_data['Passenger Fare']),
            int(input_data['Port of Embarkation'])
        ]])

        feature_names = [
            "Sex",
            "Age",
            "Passenger Class",
            "No of Parents or Children on Board",
            "No of Siblings or Spouses on Board",
            "Passenger Fare",
            "Port of Embarkation"
        ]

        # Make prediction
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        # Convert probabilities to percentages
        prob_not_survived = round(probabilities[0][0] * 100, 1)
        prob_survived = round(probabilities[0][1] * 100, 1)
        confidence_score = max(prob_not_survived, prob_survived)
        raw_prediction = int(prediction[0])

        # Generate a reason for prediction
        explanation = generate_reason_in_bahasa(features[0], raw_prediction)

        # Prepare the response
        response = {
            'model_name': model_name,
            'survived': True if raw_prediction == 1 else False,
            'raw_prediction': raw_prediction,
            'probabilities': {
                'not_survived': prob_not_survived,
                'survived': prob_survived
            },
            'confidence_score': f"{confidence_score}%",
            'alasan': explanation
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_reason_in_bahasa(features, prediction):
    """
    Generate a human-friendly explanation for the prediction in Bahasa Indonesia.
    """
    sex, age, pclass, parents, siblings, fare, embarkation = features

    reasons = []

    if prediction == 1:  # Survived
        if sex == 1:  # Female passengers
            reasons.append("Penumpang perempuan memiliki peluang lebih besar untuk selamat.")
        if pclass == 1:
            reasons.append("Anda berada di kelas satu, yang meningkatkan peluang keselamatan.")
        if fare > 50:
            reasons.append("Tarif tiket yang lebih mahal meningkatkan kemungkinan selamat.")
        if age <= 12:
            reasons.append("Anak-anak cenderung lebih diprioritaskan untuk diselamatkan.")
        if parents > 0 or siblings > 0:
            reasons.append("Perjalanan dengan keluarga meningkatkan kemungkinan dukungan di saat darurat.")
    else:  # Not survived
        if sex == 0:  # Male passengers
            reasons.append("Sebagai penumpang laki-laki, peluang selamat lebih rendah dibanding perempuan.")
        if pclass >= 3:
            reasons.append("Berada di kelas ekonomi (kelas tiga) mengurangi peluang keselamatan.")
        if fare <= 10:
            reasons.append("Tarif tiket yang murah sering dikaitkan dengan peluang selamat yang lebih kecil.")
        if age > 50:
            reasons.append("Penumpang berusia lanjut memiliki risiko lebih tinggi dalam keadaan darurat.")
        if parents == 0 and siblings == 0:
            reasons.append("Bepergian sendirian membuat Anda kurang mendapat dukungan saat keadaan darurat.")

    # Combine reasons into a single string
    if reasons:
        return " ".join(reasons)
    else:
        return "Prediksi ini dibuat berdasarkan data yang Anda berikan."


    
@app.route('/data-summary', methods=['GET'])
def data_summary():
    try:
        gender = titanic_df['Sex'].value_counts().to_dict()

        survived = titanic_df['Survived'].value_counts().to_dict()

        passenger_class = titanic_df['Passenger Class'].value_counts().to_dict()

        port_of_embarkation = titanic_df['Port of Embarkation'].value_counts().to_dict()

        return jsonify({
            "gender": gender,
            "passenger_class": passenger_class,
            "passenger_fare": {
                "median": titanic_df['Passenger Fare'].median(),
                "mean": titanic_df['Passenger Fare'].mean(),
                "highest": titanic_df['Passenger Fare'].max(),
                "lowest": titanic_df['Passenger Fare'].min()
            },
            "parents_children": {
                "median": titanic_df['No of Parents or Children on Board'].median(),
                "mean": titanic_df['No of Parents or Children on Board'].mean(),
                "highest": titanic_df['No of Parents or Children on Board'].max(),
                "lowest": titanic_df['No of Parents or Children on Board'].min()
            },
            "siblings_spouses": {
                "median": titanic_df['No of Siblings or Spouses on Board'].median(),
                "mean": titanic_df['No of Siblings or Spouses on Board'].mean(),
                "highest": titanic_df['No of Siblings or Spouses on Board'].max(),
                "lowest": titanic_df['No of Siblings or Spouses on Board'].min()
            },
            "port_of_embarkation": port_of_embarkation,
            "survived": survived
        }), 200
    except Exception as e:
        return jsonify({
            "message": str(e)
        }), 500

@app.route('/data-overview', methods=['GET'])
def data_overview():
    """
    Return general information about the dataset: 
    shape, columns, missing values, and summary stats.
    """
    overview = {
        "shape": titanic_df.shape,
        "columns": titanic_df.columns.tolist(),
        "missing_values": titanic_df.isnull().sum().to_dict(),
        "describe": titanic_df.describe(include='all').fillna('null').to_dict()
    }
    return jsonify(overview)

@app.route('/plot/age-distribution', methods=['GET'])
def plot_age_distribution():
    """Plot the age distribution of passengers."""
    plt.figure(figsize=(8, 6))
    sns.histplot(titanic_df['Age'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title("Age Distribution of Passengers")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    
    # Save plot to memory
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/plot/survival-rate-gender', methods=['GET'])
def plot_survival_rate_gender():
    plt.figure(figsize=(8, 6))
    titanic_df['Survived'] = titanic_df['Survived'].map({ 'Yes': 1, 'No': 0 })
    survival_rate = titanic_df.groupby('Sex')['Survived'].mean()
    survival_rate.plot(kind='bar', color=['lightcoral', 'skyblue'])
    plt.title("Survival Rate by Gender")
    plt.ylabel("Survival Rate")
    plt.xlabel("Gender")
    plt.xticks(rotation=0)
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/plot/fare-distribution', methods=['GET'])
def plot_fare_distribution():
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Passenger Fare', y='Passenger Class', data=titanic_df, palette="coolwarm")
    plt.title("Fare Distribution by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare")
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/plot/correlation-heatmap', methods=['GET'])
def plot_correlation_heatmap():
    plt.figure(figsize=(8, 6))

    titanic_df['Survived'] = titanic_df['Survived'].map({'Yes': 1, 'No': 0})
    titanic_df['Sex'] = titanic_df['Sex'].map({'Female': 1, 'Male': 0})
    titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
    titanic_df['Passenger Class'] = titanic_df['Passenger Class'].map({'First': 1, 'Second': 2, 'Third': 3})
    titanic_df['Passenger Fare'] = pd.to_numeric(titanic_df['Passenger Fare'], errors='coerce')
    titanic_df['Passenger Fare'] = titanic_df['Passenger Fare'].fillna(titanic_df['Passenger Fare'].median())
    titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].fillna(titanic_df['Port of Embarkation'].mode()[0])
    port_mapping = {'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3}
    titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].map(port_mapping)
    titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].fillna(0)

    corr_column = ['Sex', 'Age', 'Passenger Class', 'No of Parents or Children on Board', 'No of Siblings or Spouses on Board', 'Passenger Fare', 'Port of Embarkation', 'Survived']

    sns.heatmap(titanic_df[corr_column].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
