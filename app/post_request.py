import requests, pickle
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from src.client import Client
from src.postprocess import Postprocess

from PIL import Image
import io

## user defined types for type hinting support
GenericNumber = Union[int, float]
Key, Value = Union[str, int], Union[str, GenericNumber]
Prediction = Dict[Key, Value] # singular prediction
Detections = Dict[str, List[Prediction]] # list of predicted objects detected per image
IndexMap = List[Union[str, int]]
BytesFiles = List[Tuple[str, bytes]]

def load_df_from_csv(file):
    df = pd.read_csv(file)
    # Create a new DataFrame 'df_10' containing the first 10 rows
    df_10 = df.iloc[0:10]
    data_dict = df_10.to_dict(orient='index')
    selected_categories = ['Street', 'City', 'State', 'ZipCode']
    filtered_dict = {index: {category: data_dict[index][category] for category in selected_categories} for index in data_dict}
    return filtered_dict

def post_files(df_dict) -> Tuple[IndexMap, BytesFiles]: # Tuple[List[str],BytesFiles]
    """ 
    incoming_request_files to post in request
    """
    # df_dict needs to be in this format with index of observations as keys
    # test1 = dict(street='2301 BEAMREACH CT', city='lincoln', state='CA', zipcode='95648')
    # test2 = dict(street='3731 CEDARGATE WAY', city='SACRAMENTO', state='CA')
    img_bytes = [] # list of bytes objects
    for n in df_dict.values():
        client = Client(**n)
        response = client.response()
        img_bytes.append(response.content) # appends bytes content of an image to img_bytes list
    # img_bw = Image.open(io.BytesIO(img_bytes[6]))
    # img = img_bw.convert('RGB')
    # img.show()
    return (
        list(df_dict.keys()), 
        [('files', file) for file in img_bytes]
        )

def request_model(df_dict) -> Dict[str, Union[IndexMap, Detections]]:
    files = post_files(df_dict)
    data = dict(index=files[0]) # df unique id number
    multiple_files = files[1]
    response = requests.post('http://127.0.0.1:5000/', files=multiple_files, data=data)
    return response

def postprocess_predictions(response):
    postprocess = Postprocess()
    response_json = response.json() 
    postprocess.detection_map = (response_json['index'], response_json['detections'])
    obj_index_map = postprocess.updated_detections() # predictions mapped back to address's unique ID
    obj_count = postprocess.object_count()
    return obj_index_map, obj_count

def update(root):
    # take cmd line args, i.e. path to csv file
    # then call functions in the order below:
    print(root)
    df_dict = load_df_from_csv(root)
    print(df_dict)
    response = request_model(df_dict)
    obj_index_map, obj_count = postprocess_predictions(response)
    df = pd.read_csv(root)  # Read the CSV file without setting the index column
    # Iterate through the dictionary and update the DataFrame
    for key, value in obj_count.items():
        if value[0][0] > 0:
            df.at[key, 'SolarPanels'] = True
        else:
            df.at[key, 'SolarPanels'] = False
        
        if value[0][1] > 0:
            df.at[key, 'Pool'] = True
        else:
            df.at[key, 'Pool'] = False

    # Reset the index before saving the updated DataFrame
    df = df.reset_index(drop=True)

    # Save the updated DataFrame
    df.to_csv(root, index=False)



if __name__ == '__main__':
    update('static/csv/address_20.csv')
    