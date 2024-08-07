from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/credit_scoring_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.form.to_dict()

        # Convert the data into a DataFrame
        df = pd.DataFrame(data, index=[0])

        # Convert string values to appropriate types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Ensure the DataFrame has all the required features in the correct order
        required_features = ['UnsecLines', 'age', 'Late3059', 'DebtRatio', 'MonthlyIncome', 
                             'OpenCredit', 'Late90', 'PropLines', 'Late6089', 'Deps']
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0  # Or any appropriate default value

        df = df[required_features]

        # Make prediction
        prediction = model.predict_proba(df)[:, 1][0]

        # Return the result
        return render_template('result.html', probability=f"{prediction*100:.2f}")

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)