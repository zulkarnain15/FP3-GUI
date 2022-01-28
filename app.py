from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__) 
model = pickle.load(open('models/rf_model.pkl', 'rb'))
scaler_ = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
   
    age = int(request.form['age'])
    c_p = float(request.form['c_p'])
    e_f = float(request.form['e_f'])
    plate = float(request.form['plate'])
    s_creatinine = float(request.form['s_creatinine'])
    s_sodium = float(request.form['s_sodium'])
    time = int(request.form['time'])

    val = [age, c_p, e_f, plate, s_creatinine, s_sodium, time]
    val = scaler_.transform([val])
    val_predict = model.predict(val)
    return render_template('predict.html', data=val_predict)

if __name__ == "__main__":
    app.run(debug=True)