from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('models/rf_model.pkl')  # Use the path to your saved model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    
    # Create DataFrame
    feature_names = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                     'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    df = pd.DataFrame([features], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    
    result = "Default" if prediction[0] == 1 else "No Default"
    
    return render_template('result.html', prediction=result, probability=round(probability*100, 2))

if __name__ == '__main__':
    app.run(debug=True)