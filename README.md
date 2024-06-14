# Wine Quality Predictor

The Wine Quality Predictor is a web application built using Flask that predicts the quality of a wine based on various chemical properties. The application trains a machine learning model on a wine quality dataset and uses the trained model to make predictions.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

- Python 
- Flask
- scikit-learn
- joblib
- pandas
- numpy
- xgboost

### Installation

1. Clone the repository:
git clone https://github.com/sanjaykohli/Wine-Quality-Predictor.git
  
2. Navigate to the project directory:
cd Wine-Quality-Predictor

3. Install the required packages:
pip install -r requirements.txt

4. Run the Flask application:
python app.py

5. Open your web browser and visit `http://localhost:5000` to access the Wine Quality Predictor.

## Usage

1. Enter the required wine properties in the input fields:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

2. Click the "Predict" button.
3. The application will display the predicted quality of the wine as "High Quality" or "Low Quality".
4. If any input field is left blank or contains invalid data, an error message will be displayed, prompting the user to provide valid inputs.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
