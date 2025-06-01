from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/standscaler.pkl", "rb"))
minmax = pickle.load(open("models/minmaxscaler.pkl", "rb"))
label_encoder = pickle.load(open("models/labelencoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_values = [float(request.form.get(k)) for k in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    input_np = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(minmax.transform(input_np))
    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)[0]
    return render_template('index.html', result=f"Recommended crop: {result}")

if __name__ == '__main__':
    app.run(debug=True, port=8001)
