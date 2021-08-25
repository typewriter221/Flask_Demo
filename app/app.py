from flask import Flask, render_template, request
from flask.wrappers import Request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import os 
import uuid
app = Flask(__name__)
model = load('model.joblib')

TRAINING_DATA = "AgesAndHeights.pkl"
@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    
    request_type = request.method

    if request_type == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    if request_type == 'POST':
        input = request.form['ages']
        inputs = string_to_float(input)
        random_string = uuid.uuid4().hex
        OUT_PATH = 'static/'+random_string+'.svg'
        make_picture(TRAINING_DATA, model, inputs,OUT_PATH )
        
        return render_template('index.html', href = OUT_PATH)

    # model = load('model.joblib')
    # input = [[20],[30]]
    # pred = model.predict(input)


def make_picture(training_data, model, inputs, OUT_PATH):
    inputs = np.array(inputs)
    data = pd.read_pickle(training_data)
    data = data[data['Age']>0]
    heights = data['Height'].to_numpy()
    ages = data['Age'].to_numpy()
    x_rand = np.array(list(range(19)))
    pred = model.predict(x_rand.reshape(-1,1))
    fig = px.scatter(x = ages, y = heights, title = 'Height V/S Age', labels = {'x':'Age (years)', 'y':"Height (inches)"} )
    fig.add_trace(go.Scatter(mode='lines', x = x_rand, y = pred.reshape(-1), name = 'Model'))
    new_points = model.predict(inputs.reshape(-1,1))
    fig.add_trace(go.Scatter(mode='markers', x = inputs, y = new_points.reshape(-1), name = 'New Points', marker = dict(color = 'purple', size = 20, line = dict(color = 'purple', width = 2))))

    fig.write_image(OUT_PATH, width = 800, engine = 'kaleido')
    
    
    fig.show()


def string_to_float(st):
    values = []
    for s in st.split(','):
        try:
            f = float(s)
            values.append(f)
        except:
            pass
    return np.array(values)
