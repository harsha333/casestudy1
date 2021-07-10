from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
def feature_engineering(df):
    df['mag_orientation']=np.sqrt(df['orientation_X']**2 + df['orientation_Y']**2 + df['orientation_Z']**2 + df['orientation_W']**2)
    df['mag_angular_velocity']=np.sqrt(df['angular_velocity_X']**2 + df['angular_velocity_Y']**2 + df['angular_velocity_Z']**2)
    df['mag_linear_acceleration']=np.sqrt(df['linear_acceleration_X']**2 + df['linear_acceleration_Y']**2 + df['linear_acceleration_Z']**2)
    return df


def data_preparation(data):
    data = np.array(data)
    print(data.shape)
    data = data.reshape(1, 11)
    print(data.shape)
    query_point_df = pd.DataFrame()
    query_point_df = pd.DataFrame(data, columns=['series_id', 'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W', 'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z'])
    query_point_df = feature_engineering(query_point_df)
    return query_point_df
###################################################


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    dict_labels = {0: 'carpet', 1: 'concrete', 2: 'fine_concrete', 3: 'hard_tiles', 4: 'hard_tiles_large_space', 5: 'soft_pvc', 6: 'soft_tiles', 7: 'tiled', 8: 'wood'}
    clf = joblib.load('best_model_RF.pkl')
    to_predict_list = request.form.to_dict()
    sensor_data = to_predict_list['sensor_data']
    sensor_data = sensor_data.split(',')
    sensor_dta = [float(x) for x in sensor_data]
    print(sensor_dta)
    query_point_df = data_preparation(sensor_dta)
    y_qp_pred = int(clf.predict(query_point_df))
    prediction = dict_labels[y_qp_pred]

    return jsonify({'prediction': prediction})
