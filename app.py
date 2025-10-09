import os
import json
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template

# --- Firestore Imports and Configuration ---
try:
    from firebase_admin import credentials, initialize_app, firestore
except ImportError:
    print("WARNING: firebase-admin not installed. Firestore features will be skipped.")
    firestore = None

# Flask setup
app = Flask(__name__)

# --- Environment and Database Setup ---

__app_id = os.environ.get('APP_ID', 'default-app-id') 

FEEDBACK_COLLECTION_PATH = f'artifacts/{__app_id}/public/data/spam_feedback'

db = None
try:
    service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

    if service_account_json and firestore:
        cred_dict = json.loads(service_account_json)
        
        project_id = cred_dict.get('project_id') 

        if not project_id:
            raise ValueError("Service Account JSON is missing 'project_id'.")
        
        database_url = f'https://{project_id}.firebaseio.com' 
        
        cred = credentials.Certificate(cred_dict)
        initialize_app(cred, options={
            'projectId': project_id,
            'databaseURL': database_url 
        })
        
        db = firestore.client()
        print(f"INFO: Firebase Admin SDK initialized. Logging to collection: {FEEDBACK_COLLECTION_PATH}")
    else:
        print("WARNING: Firestore logging disabled (Missing credentials or firebase-admin).")
        
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize Firebase Admin SDK. Check FIREBASE_SERVICE_ACCOUNT_JSON format. Error: {e}")
    db = None

# --- Model Loading ---

MODEL_PATH = 'spam_detection_pipeline_V2.pkl'

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model_pipeline = joblib.load(file)
    print(f"INFO: Model loaded successfully from {MODEL_PATH}")
else:
    print(f"ERROR: Model file {MODEL_PATH} not found. Prediction will not work.")
    model_pipeline = None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    if not model_pipeline:
        return jsonify({'prediction': 'ham', 'message': 'Model unavailable.', 'error': True}), 500

    if not message:
        return jsonify({'prediction': 'ham', 'message': 'Please enter a message.'}), 400

    prediction_result = model_pipeline.predict([message])
    prediction = prediction_result[0]

    return jsonify({
        'prediction': prediction,
        'message': 'Prediction successful.',
        'error': False
    })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Receives user feedback and logs it to Firestore."""
    data = request.get_json()
    
    message = data.get('message')
    tool_prediction = data.get('toolPrediction')
    correct_label = data.get('correctLabel')
    
    if not all([message, tool_prediction, correct_label]):
        return jsonify({'status': 'error', 'message': 'Missing required feedback fields'}), 400

    # 1. Structure the data
    feedback_data = {
        'message': message,
        'tool_prediction': tool_prediction,
        'correct_label': correct_label,
        'timestamp': datetime.now(),
        'app_id': __app_id
    }

    # 2. Write to Firestore
    if db:
        try:
            # FIX: We now simply call .add() to write the data.
            # We don't try to capture the Document ID for local printing,
            # which completely bypasses the problematic tuple unpacking issue.
            db.collection(FEEDBACK_COLLECTION_PATH).add(feedback_data)
            
            # The log is now a simpler confirmation, removing the .id access
            print(f"LOGGED: Feedback saved to Firestore.") 
            return jsonify({'status': 'success', 'message': 'Feedback received and logged to Firestore.'})
        except Exception as e:
            print(f"FIREBASE WRITE ERROR: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to log to Firestore: {e}'}), 500
    else:
        return jsonify({'status': 'warning', 'message': 'Feedback not logged: Firestore is not initialized.'}), 200

if __name__ == '__main__':
    app.run(debug=True)
