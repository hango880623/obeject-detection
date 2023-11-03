from flask import Flask, request, jsonify, render_template, redirect, url_for
from src.model_utils import Yolo
import pandas as pd
from post_request import update
import os
import time

app = Flask(__name__)

# Load your DataFrame here
model_path = 'static/best1013.pt'
base_path = 'static/csv/'
yolo_model = Yolo(weights_=model_path)
file_path = os.path.join(base_path,'address_20.csv')
df = pd.read_csv(file_path)

def update_file_path_variable(new_file):
    global file_path
    file_path = os.path.join(base_path,new_file)
    global df
    df = pd.read_csv(file_path)

@app.route('/', methods=['POST'])
def batch_request():
    if request.method == 'POST':
        files = request.files.getlist('files')
        post_request_data = request.form.to_dict(flat=False)['index']
        if not files:
            return
        else:
            yolo_model.files = files
            predictions = yolo_model.get_predictions() # list of predictions
        detection_map = dict(index=post_request_data, detections=predictions)
        return jsonify(detection_map)
    
@app.route('/', methods=['GET'])
def display_df():
    global df  # Ensure the function uses the global DataFrame
    # Convert the DataFrame to HTML
    table_html = df.to_html(classes='table table-bordered table-striped', index=False, escape=False)
    return render_template('display.html', table_html=table_html)

@app.route('/predict_data', methods=['POST'])
def predict_data():
    update(file_path)  # Call the 'update()' function to update the DataFrame
    return redirect(url_for('display_df'))  # Return a message indicating the update was successful

@app.route('/static_files_json', methods=['GET'])
def list_static_files_json():
    static_dir = 'static/csv'
    static_files = os.listdir(static_dir)
    return jsonify(static_files=static_files)

@app.route('/update_file_path', methods=['POST'])
def update_file_path():
    new_file_path = request.form.get('new_file_path')  # Get the value from the "new_file_path" input
    if new_file_path:
        update_file_path_variable(new_file_path)  # Update the file_path
        return jsonify(success=True)
    else:
        return jsonify(success=False)

if __name__ == '__main__':
    app.run(debug=True)