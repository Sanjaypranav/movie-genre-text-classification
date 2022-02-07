'''webserver gateway run 2 methods Get() Post() are main methods for flask '''
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import jsonify
import json

app = Flask(__name__)

vectorizer = pickle.load(open('Models/vectorizer.pkl', 'rb'))
labelbinarizer = pickle.load(open('Models/target_label.pkl', 'rb'))
model = pickle.load(open('Models/movie_model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        data = request.form['message']
        print("data = ",data)
        print(model.predict(vectorizer.transform([data]).toarray()))
        predicted = labelbinarizer.inverse_transform(model.predict(vectorizer.transform([data]).toarray()))
        print("Prediction = ",predicted)
    return render_template('index.html', prediction="Expected genres are ===> {}".format(predicted))

if __name__ == '__main__':
    app.run(debug=True)

#file