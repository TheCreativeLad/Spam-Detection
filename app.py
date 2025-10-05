import os
import json
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template

# --- Firestore Imports and Configuration ---
# Note: These imports use the firebase-admin SDK for server-side access
try:
    from firebase_admin import credentials, initialize_app, firestore
except ImportError:
    print("WARNING: firebase-admin not installed. Firestore features will be skipped.")
    firestore = None # Set to None if import fails for local dev testing

# Flask setup
app = Flask(__name__)

# --- Environment and Database Setup ---

# Global variables (provided by Canvas environment, mocked for local test)
__app_id = os.environ.get('APP_ID', 'default-app-id') 

# Define the collection path where feedback will be stored
# This follows the public data path security rules.
FEEDBACK_COLLECTION_PATH = f'artifacts/{__app_id}/public/data/spam_feedback'

# Initialize Firestore Client
db = None
try:
    # Render Secret: The entire JSON file content is stored in this variable.
    service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

    if service_account_json and firestore:
        # 1. Parse the JSON string into a Python dictionary
        cred_dict = json.loads(service_account_json)
        
        # 2. Use the dictionary content to initialize the Admin SDK
        cred = credentials.Certificate(cred_dict)
        initialize_app(cred)
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
    # Load the trained model pipeline
    with open(MODEL_PATH, 'rb') as file:
        model_pipeline = joblib.load(file)
    print(f"INFO: Model loaded successfully from {MODEL_PATH}")
else:
    print(f"ERROR: Model file {MODEL_PATH} not found. Prediction will not work.")
    model_pipeline = None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main prediction interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives a message and returns the spam/ham prediction."""
    # This route now expects a JSON payload from the frontend
    data = request.get_json()
    message = data.get('message', '')

    if not model_pipeline:
        return jsonify({'prediction': 'ham', 'message': 'Model unavailable.', 'error': True}), 500

    if not message:
        return jsonify({'prediction': 'ham', 'message': 'Please enter a message.'}), 400

    # The pipeline handles tokenization, feature extraction, and prediction
    prediction_result = model_pipeline.predict([message])
    
    # The output is an array, so we take the first element (e.g., 'spam' or 'ham')
    prediction = prediction_result[0]

    return jsonify({
        'prediction': prediction,
        'message': 'Prediction successful.',
        'error': False
    })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Receives user feedback and logs it to Firestore."""
    # This route now expects a JSON payload from the frontend
    data = request.get_json()
    
    message = data.get('message')
    tool_prediction = data.get('toolPrediction')
    correct_label = data.get('correctLabel')
    
    if not all([message, tool_prediction, correct_label]):
        return jsonify({'status': 'error', 'message': 'Missing required feedback fields'}), 400

    # 1. Structure the data
    feedback_data = {
        'message': message,
        'tool_prediction': tool_prediction,  # What V2 thought (e.g., 'ham')
        'correct_label': correct_label,      # What the user corrected it to (e.g., 'spam')
        'timestamp': datetime.now(),
        'app_id': __app_id
    }

    # 2. Write to Firestore
    if db:
        try:
            # Add a new document to the collection
            # Using .add() generates a unique document ID automatically
            doc_ref, _ = db.collection(FEEDBACK_COLLECTION_PATH).add(feedback_data)
            print(f"LOGGED: Feedback saved to Firestore. Document ID: {doc_ref.id}")
            return jsonify({'status': 'success', 'message': 'Feedback received and logged to Firestore.'})
        except Exception as e:
            print(f"FIREBASE WRITE ERROR: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to log to Firestore: {e}'}), 500
    else:
        # This fallback happens if initialization failed
        return jsonify({'status': 'warning', 'message': 'Feedback not logged: Firestore is not initialized.'}), 200

if __name__ == '__main__':
    # When running locally, Flask runs on the default port 5000
    app.run(debug=True)
