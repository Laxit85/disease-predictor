from flask import Flask, render_template, request, jsonify
from ml import predict_disease

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    result = predict_disease(symptoms)
    if isinstance(result, str):  # error message
        return jsonify({'error': result}), 400
    else:
        predictions = [{'disease': disease, 'probability': float(prob)} for disease, prob in result]
        return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
