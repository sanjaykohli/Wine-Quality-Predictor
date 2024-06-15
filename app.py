from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

pipeline = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    col_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH','sulphates', 'alcohol']
    
    features = [float(x) for x in request.form.values()]
    
    if len(features) != len(col_names):
        return render_template('index.html', prediction_text='Invalid input: Please provide all features.')
    
    df = pd.DataFrame([features], columns=col_names)
    
    try:
        prediction = pipeline.predict(df)
        output = "High Quality" if prediction[0] == 1 else "Low Quality"
        return render_template('index.html', prediction_text=f'The wine is predicted to be: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Prediction error: {str(e)}')


