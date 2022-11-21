from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('trainedModel.sav', 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/details')
def details():
    return render_template('details.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    final_features = scaler.transform(features)
    prediction = model.predict(features)

    output = prediction

    if output == 1:
        return render_template('details.html', prediction_text='80% chance! You Have Diabetes')
    else:
        return render_template('details.html', prediction_text='80% chance! You Dont Have Diabetes')


if __name__ == '__main__':
    app.run(debug=True)
