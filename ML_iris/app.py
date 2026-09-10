from flask import Flask, render_template, request, jsonify  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

app = Flask(__name__)

# Load trained models and scaler
try:
    svm_model = joblib.load('models/svm_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please run iris_model_training.ipynb first to train and save the models.")
    svm_model = None
    knn_model = None
    scaler = None

# Feature names matching dataset
target_names = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page with form and information"""
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if models are loaded
        if svm_model is None or knn_model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please train models first.'
            }), 500
        
        # Get data with dataset column names
        SepalLengthCm = None
        SepalWidthCm = None
        PetalLengthCm = None
        PetalWidthCm = None
        model_choice = None
        
        # Try to get data from form
        if request.form:
            SepalLengthCm = request.form.get('SepalLengthCm')
            SepalWidthCm = request.form.get('SepalWidthCm')
            PetalLengthCm = request.form.get('PetalLengthCm')
            PetalWidthCm = request.form.get('PetalWidthCm')
            model_choice = request.form.get('model_choice')
            print(f"📝 Form data received: SL={SepalLengthCm}, SW={SepalWidthCm}, PL={PetalLengthCm}, PW={PetalWidthCm}, Model={model_choice}")
        
        # Try to get data from JSON
        if request.is_json:
            data = request.get_json()
            SepalLengthCm = data.get('SepalLengthCm')
            SepalWidthCm = data.get('SepalWidthCm')
            PetalLengthCm = data.get('PetalLengthCm')
            PetalWidthCm = data.get('PetalWidthCm')
            model_choice = data.get('model_choice')
            print(f"📝 JSON data received: SL={SepalLengthCm}, SW={SepalWidthCm}, PL={PetalLengthCm}, PW={PetalWidthCm}, Model={model_choice}")
        
        # Check if we got all required fields
        if None in [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, model_choice]:
            missing = []
            if SepalLengthCm is None:
                missing.append('SepalLengthCm')
            if SepalWidthCm is None:
                missing.append('SepalWidthCm')
            if PetalLengthCm is None:
                missing.append('PetalLengthCm')
            if PetalWidthCm is None:
                missing.append('PetalWidthCm')
            if model_choice is None:
                missing.append('model_choice')
            
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing)}'
            }), 400
        
        # Convert to float
        try:
            SepalLengthCm = float(SepalLengthCm)
            SepalWidthCm = float(SepalWidthCm)
            PetalLengthCm = float(PetalLengthCm)
            PetalWidthCm = float(PetalWidthCm)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid number format: {str(e)}'
            }), 400
        
        # Create feature array
        features = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction based on model choice
        if model_choice == 'svm':
            prediction = svm_model.predict(features_scaled)
            pred_class = int(prediction[0])
            
            # Handle SVM probabilities properly
            try:
                decision = svm_model.decision_function(features_scaled)
                if len(decision.shape) == 1:
                    exp_decision = np.exp(decision - np.max(decision))
                    probabilities = exp_decision / np.sum(exp_decision)
                    if len(probabilities) == 1:
                        probs = np.zeros(3)
                        probs[pred_class] = 1.0
                        probabilities = probs
                else:
                    exp_decision = np.exp(decision[0] - np.max(decision[0]))
                    probabilities = exp_decision / np.sum(exp_decision)
            except Exception:
                probabilities = np.array([0.33, 0.33, 0.34])
            
            model_name = 'Support Vector Machine (SVM)'
            
        elif model_choice == 'knn':
            prediction = knn_model.predict(features_scaled)
            pred_class = int(prediction[0])
            probabilities = knn_model.predict_proba(features_scaled)[0]
            model_name = 'K-Nearest Neighbors (KNN)'
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid model choice: {model_choice}'
            }), 400
        
        predicted_class = target_names[pred_class]
        
        # Create probability distribution
        prob_dist = {
            'Setosa': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            'Versicolor': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            'Virginica': float(probabilities[2]) if len(probabilities) > 2 else 0.0
        }
        
        print(f"✅ Prediction: {predicted_class}")
        print(f"📊 Probabilities: {prob_dist}")
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'probabilities': prob_dist,
            'model_used': model_name,
            'features': {
                'SepalLengthCm': SepalLengthCm,
                'SepalWidthCm': SepalWidthCm,
                'PetalLengthCm': PetalLengthCm,
                'PetalWidthCm': PetalWidthCm
            }
        })
    
    except Exception as e:
        print(f"❌ Error in predict: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/compare_models', methods=['POST'])
def compare_models():
    try:
        # Check if models are loaded
        if svm_model is None or knn_model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please train models first.'
            }), 500
        
        # Get JSON data
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        SepalLengthCm = float(data['SepalLengthCm'])
        SepalWidthCm = float(data['SepalWidthCm'])
        PetalLengthCm = float(data['PetalLengthCm'])
        PetalWidthCm = float(data['PetalWidthCm'])
        
        # Create feature array
        features = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
        features_scaled = scaler.transform(features)
        
        # Get predictions from both models
        svm_pred = int(svm_model.predict(features_scaled)[0])
        knn_pred = int(knn_model.predict(features_scaled)[0])
        
        # Get SVM probabilities
        try:
            svm_decision = svm_model.decision_function(features_scaled)
            if len(svm_decision.shape) == 1:
                exp_decision = np.exp(svm_decision - np.max(svm_decision))
                svm_probs = exp_decision / np.sum(exp_decision)
                if len(svm_probs) == 1:
                    svm_probs_array = np.zeros(3)
                    svm_probs_array[svm_pred] = 1.0
                    svm_probs = svm_probs_array
            else:
                exp_decision = np.exp(svm_decision[0] - np.max(svm_decision[0]))
                svm_probs = exp_decision / np.sum(exp_decision)
        except Exception:
            svm_probs = np.array([0.33, 0.33, 0.34])
        
        # Get KNN probabilities
        knn_probs = knn_model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'svm': {
                'prediction': target_names[svm_pred],
                'probabilities': {
                    'Setosa': float(svm_probs[0]) if len(svm_probs) > 0 else 0.0,
                    'Versicolor': float(svm_probs[1]) if len(svm_probs) > 1 else 0.0,
                    'Virginica': float(svm_probs[2]) if len(svm_probs) > 2 else 0.0
                }
            },
            'knn': {
                'prediction': target_names[knn_pred],
                'probabilities': {
                    'Setosa': float(knn_probs[0]),
                    'Versicolor': float(knn_probs[1]),
                    'Virginica': float(knn_probs[2])
                }
            }
        })
    
    except Exception as e:
        print(f"❌ Error in compare_models: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)