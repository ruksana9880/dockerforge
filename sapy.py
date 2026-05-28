from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

class CVDPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_info = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model = joblib.load('cvd_model.pkl')
            self.scaler = joblib.load('cvd_scaler.pkl')
            self.feature_names = joblib.load('feature_names.pkl')
            self.model_info = joblib.load('model_info.pkl')
            print("Model loaded successfully!")
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Please run train_model.py first to generate the model files.")
    
    def predict(self, input_data):
        """Make prediction on input data"""
        if self.model is None:
            return None, None
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data], columns=self.feature_names)
        
        # Scale the data if needed
        if self.model_info['model_name'] in ['Logistic Regression', 'SVM']:
            df_scaled = self.scaler.transform(df)
            prediction = self.model.predict(df_scaled)[0]
            probability = self.model.predict_proba(df_scaled)[0]
        else:
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0]
        
        return prediction, probability
    
    def get_risk_interpretation(self, probability):
        """Interpret the risk probability"""
        prob_positive = probability[1]
        
        if prob_positive < 0.3:
            return "Low Risk", "green"
        elif prob_positive < 0.6:
            return "Moderate Risk", "orange"
        else:
            return "High Risk", "red"

# Initialize predictor
predictor = CVDPredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features in the correct order
        input_features = [
            int(data['age']),
            int(data['gender']),
            int(data['systolic_bp']),
            int(data['diastolic_bp']),
            float(data['total_cholesterol']),
            float(data['ldl_cholesterol']),
            float(data['hdl_cholesterol']),
            float(data['triglycerides']),
            float(data['glucose']),
            float(data['hba1c']),
            float(data['crp']),
            float(data['troponin']),
            float(data['bnp']),
            float(data['homocysteine']),
            int(data['smoking']),
            int(data['family_history']),
            float(data['bmi'])
        ]
        
        # Make prediction
        prediction, probability = predictor.predict(input_features)
        
        if prediction is None:
            return jsonify({
                'error': 'Model not loaded. Please run train_model.py first.'
            }), 500
        
        # Get risk interpretation
        risk_level, risk_color = predictor.get_risk_interpretation(probability)
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': {
                'no_cvd': float(probability[0]),
                'cvd': float(probability[1])
            },
            'risk_level': risk_level,
            'risk_color': risk_color,
            'percentage': round(probability[1] * 100, 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'model_type': predictor.model_info['model_name'] if predictor.model_info else None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/info')
def model_info():
    """Get model information"""
    if predictor.model_info:
        return jsonify(predictor.model_info)
    else:
        return jsonify({'error': 'Model not loaded'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)