from flask import Flask,request,url_for,render_template
import numpy as np
import pickle

app = Flask(__name__)

model  = pickle.load(open('bestmodel.pkl','rb'))

@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['sex']
    data3 = request.form['cp']
    data4 = request.form['trestbps']
    data5 = request.form['fbs']
    data6 = request.form['restecg']
    data7 = request.form['exang']
    data8 = request.form['oldpeak']
    data9 = request.form['slope']
    data10 = request.form['ca']
    data11 = request.form['thal']
    arr = np.array([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11],dtype=float)
    arr = arr.reshape(1,-1)
    pred = model.predict(arr)
    return render_template('result.html',data=pred)
if __name__ == "__main__":
    app.run(debug=True)