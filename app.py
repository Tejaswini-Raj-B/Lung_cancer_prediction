from flask import Flask, render_template, request
import pickle
import numpy as np
from preprocess import preprocess_input

app = Flask(__name__)
model = pickle.load(open('model/lung_cancer_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form.get('Age')),
        int(request.form.get('Gender')),
        int(request.form.get('Smoking')),
        int(request.form.get('Yellow_Fingers')),
        int(request.form.get('Anxiety')),
        int(request.form.get('Peer_Pressure')),
        int(request.form.get('Chronic_Disease')),
        int(request.form.get('Fatigue')),
        int(request.form.get('Allergy')),
        int(request.form.get('Wheezing')),
        int(request.form.get('Alcohol')),
        int(request.form.get('Coughing')),
        int(request.form.get('Shortness_of_Breath')),
        int(request.form.get('Swallowing_Difficulty')),
        int(request.form.get('Chest_Pain')),
    ]

    processed = preprocess_input(features)
    prediction = model.predict([processed])[0]
    result = "Positive" if prediction == 1 else "Negative"

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
