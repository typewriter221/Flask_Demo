import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load

model = load('app/model.joblib')
def make_picture(training_data, model, inputs):
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
    # fig.add_trace(go.Scatter(mode='markers', x = inputs, y = new_points.reshape(-1), name = 'New Points', marker = dict(color = 'purple', size = 20, line = dict(color = 'purple', width = 2))))
    
    fig.write_image('output.svg', width = 800)
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

make_picture('app\AgesAndHeights.pkl', model, [15, 14, 10])