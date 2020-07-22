import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def result():

    dataset = pd.read_csv('odi_cricket_dataset.csv')
    vectorized = pd.get_dummies(dataset.iloc[:,1:4])

    flag = [False, False, False]

    record = np.zeros((1,131),dtype='float64')
    input = [x for x in request.form.values()]
    t = "Team_" + input[0]
    o = "Opposition_v\xa0" + input[1]
    g = "Ground_" + input[2]
    r = int(input[3])

    record[0,130] = r

    for i,vec in enumerate(vectorized.columns):
        if vec == t:
            record[0,i] = 1
            flag[0] = True
        if vec == o:
            record[0,i] = 1
            flag[1] = True
        if vec == g:
            record[0,i] = 1
            flag[2] = True

    if(flag[0]==True and flag[1]==True and flag[2]==True):
        filename = 'model_2.sav'
        model = pickle.load(open(filename, 'rb'))
        ans = model.predict(record)

        if ans == 1:
            return render_template('index.html', prediction_text='The {} will win the match'.format(input[0]))
        elif ans == 0:
            return render_template('index.html', prediction_text='The {} will win the match'.format(input[1]))

    else:
        return render_template('index.html', prediction_text='Incorrect Details')

if __name__ == "__main__":
    app.run(debug=True)
