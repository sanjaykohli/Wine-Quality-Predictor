# app.py

from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
pipeline = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Define the column names expected by your model
    col_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH','sulphates', 'alcohol']
    
    # Extract and convert input data into a DataFrame
    features = [float(x) for x in request.form.values()]
    
    # Check if the number of features matches the expected number of columns
    if len(features) != len(col_names):
        return render_template('index.html', prediction_text='Invalid input: Please provide all features.')
    
    # Create DataFrame from input data
    df = pd.DataFrame([features], columns=col_names)
    
    try:
        # Perform prediction using the loaded pipeline
        prediction = pipeline.predict(df)
        output = "High Quality" if prediction[0] == 1 else "Low Quality"
        return render_template('index.html', prediction_text=f'The wine is predicted to be: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Prediction error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
