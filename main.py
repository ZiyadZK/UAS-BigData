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
    """
    Return dataset summary with survival analysis based on gender, class, port, and age.
    """
    try:
        # Create subplots
        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        plt.tight_layout(pad=6)

        # Plot 1: Gender vs Survival
        gender_survived = titanic_df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
        gender_survived.plot(kind='bar', stacked=True, ax=axs[0, 0], color=['red', 'green'])
        axs[0, 0].set_title('Survival Based on Gender')
        axs[0, 0].set_xlabel('Gender')
        axs[0, 0].set_ylabel('Count')

        # Plot 2: Passenger Class vs Survival
        class_survived = titanic_df.groupby(['Passenger Class', 'Survived']).size().unstack(fill_value=0)
        class_survived.plot(kind='bar', stacked=True, ax=axs[0, 1], color=['red', 'green'])
        axs[0, 1].set_title('Survival Based on Passenger Class')
        axs[0, 1].set_xlabel('Passenger Class')
        axs[0, 1].set_ylabel('Count')

        # Plot 3: Port of Embarkation vs Survival
        port_survived = titanic_df.groupby(['Port of Embarkation', 'Survived']).size().unstack(fill_value=0)
        port_survived.plot(kind='bar', stacked=True, ax=axs[1, 0], color=['red', 'green'])
        axs[1, 0].set_title('Survival Based on Port of Embarkation')
        axs[1, 0].set_xlabel('Port of Embarkation')
        axs[1, 0].set_ylabel('Count')

        # Plot 4: Age Distribution
        titanic_df['Age Group'] = pd.cut(titanic_df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                         labels=['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior'])
        age_survived = titanic_df.groupby(['Age Group', 'Survived']).size().unstack(fill_value=0)
        age_survived.plot(kind='bar', stacked=True, ax=axs[1, 1], color=['red', 'green'])
        axs[1, 1].set_title('Survival Based on Age Group')
        axs[1, 1].set_xlabel('Age Group')
        axs[1, 1].set_ylabel('Count')

        # Plot 5: Overall Survival Count
        survived_counts = titanic_df['Survived'].value_counts()
        axs[2, 0].bar(['Not Survived', 'Survived'], survived_counts.values, color=['red', 'green'])
        axs[2, 0].set_title('Overall Survival Count')
        axs[2, 0].set_xlabel('Survival Status')
        axs[2, 0].set_ylabel('Count')

        # Plot 6: Fare Distribution
        axs[2, 1].hist(titanic_df['Passenger Fare'], bins=20, color='purple', edgecolor='black')
        axs[2, 1].set_title('Fare Distribution')
        axs[2, 1].set_xlabel('Fare')
        axs[2, 1].set_ylabel('Frequency')

        # Save the plots as a single image
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        return send_file(img, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/data/age-distribution', methods=['GET'])
def age_distribution_chart():
    """Return an age distribution chart."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot the age distribution
        sns.histplot(titanic_df['Age'].dropna(), bins=30, kde=True, color='purple')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data/survival-rate-gender', methods=['GET'])
def survival_rate_by_gender_chart():
    """Return a bar chart for survival rates by gender."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Map 'Survived' values and calculate survival rates by gender
        titanic_df['Survived'] = titanic_df['Survived'].map({'Yes': 1, 'No': 0})
        survival_rate = titanic_df.groupby('Sex')['Survived'].mean()
        survival_rate.plot(kind='bar', color=['red', 'green'], alpha=0.7)
        plt.title('Survival Rate by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        
        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data/fare-distribution', methods=['GET'])
def fare_distribution_chart():
    """Return a boxplot of fare distribution by passenger class."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot a boxplot for fare distribution grouped by passenger class
        sns.boxplot(x='Passenger Class', y='Passenger Fare', data=titanic_df, palette='Set2')
        plt.title('Fare Distribution by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Fare')
        
        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data/correlation-heatmap', methods=['GET'])
def correlation_heatmap_chart():
    """Return a correlation heatmap."""
    try:
        plt.figure(figsize=(10, 8))
        
        # Prepare data for correlation
        titanic_df['Sex'] = titanic_df['Sex'].map({'Female': 1, 'Male': 0})
        titanic_df['Survived'] = titanic_df['Survived'].map({ 'Yes': 1, 'No': 0})
        titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
        titanic_df['Passenger Class'] = titanic_df['Passenger Class'].map({'First': 1, 'Second': 2, 'Third': 3})

        titanic_df['Passenger Fare'] = pd.to_numeric(titanic_df['Passenger Fare'], errors='coerce')
        titanic_df['Passenger Fare'] = titanic_df['Passenger Fare'].fillna(titanic_df['Passenger Fare'].median())


        titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].fillna(titanic_df['Port of Embarkation'].mode()[0])


        port_mapping = {'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3}
        titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].map(port_mapping)

        titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].fillna(0)

        columns_for_correlation = ['Sex', 'Age', 'Passenger Class', 'No of Parents or Children on Board', 'No of Siblings or Spouses on Board', 'Passenger Fare', 'Port of Embarkation', 'Survived']

        corr_matrix = titanic_df[columns_for_correlation].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        
        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data/correlation-explanation', methods=['GET'])
def correlation_explanation():
    """
    Return a textual explanation of the correlation between key features.
    """
    try:
        # Ensure the dataset is properly prepared
        titanic_df['Sex'] = titanic_df['Sex'].map({'Female': 1, 'Male': 0})
        titanic_df['Survived'] = titanic_df['Survived'].map({'Yes': 1, 'No': 0})
        titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
        titanic_df['Passenger Class'] = titanic_df['Passenger Class'].map({'First': 1, 'Second': 2, 'Third': 3})

        titanic_df['Passenger Fare'] = pd.to_numeric(titanic_df['Passenger Fare'], errors='coerce')
        titanic_df['Passenger Fare'] = titanic_df['Passenger Fare'].fillna(titanic_df['Passenger Fare'].median())

        titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].fillna(titanic_df['Port of Embarkation'].mode()[0])
        port_mapping = {'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3}
        titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].map(port_mapping)
        titanic_df['Port of Embarkation'] = titanic_df['Port of Embarkation'].fillna(0)

        # Select columns for correlation analysis
        columns_for_correlation = [
            'Sex', 'Age', 'Passenger Class', 
            'No of Parents or Children on Board', 
            'No of Siblings or Spouses on Board', 
            'Passenger Fare', 'Port of Embarkation', 'Survived'
        ]

        # Compute the correlation matrix
        corr_matrix = titanic_df[columns_for_correlation].corr()

        # Extract meaningful correlations with survival
        survival_corr = corr_matrix['Survived'].sort_values(ascending=False)

        # Generate a human-readable explanation
        explanation = []
        for feature, corr_value in survival_corr.items():
            if corr_value > 0.5:
                explanation.append(
                    f"The feature '{feature}' has a strong positive correlation ({corr_value:.2f}) with survival. "
                    f"This means passengers with higher values in this feature were more likely to survive."
                )
            elif corr_value > 0.3:
                explanation.append(
                    f"The feature '{feature}' has a moderate positive correlation ({corr_value:.2f}) with survival. "
                    f"This suggests some impact on survival likelihood."
                )
            elif corr_value < -0.5:
                explanation.append(
                    f"The feature '{feature}' has a strong negative correlation ({corr_value:.2f}) with survival. "
                    f"This indicates passengers with higher values in this feature were less likely to survive."
                )
            elif corr_value < -0.3:
                explanation.append(
                    f"The feature '{feature}' has a moderate negative correlation ({corr_value:.2f}) with survival. "
                    f"This suggests some decrease in survival likelihood."
                )
            else:
                explanation.append(
                    f"The feature '{feature}' has a weak correlation ({corr_value:.2f}) with survival. "
                    f"This indicates minimal influence on survival chances."
                )

        return jsonify({
            "correlation_explanation": explanation
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
