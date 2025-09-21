from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved pipeline
# The pipeline includes the vectorizer and the model
loaded_pipeline = joblib.load('spam_detection_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = loaded_pipeline.predict([message])
        
        # Determine the prediction result
        if prediction[0] == 'spam':
            result = "This message is Spam! 😠"
        else:
            result = "This message is not Spam. ✅"
            
        return render_template('result.html', prediction=result, original_message=message)

if __name__ == '__main__':
    app.run(debug=True)