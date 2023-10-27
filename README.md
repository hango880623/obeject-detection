# Object Detection Api

- **This code still a work in progress**

In the project root directory *app/*,

- *app.py* contains a simple api that handles POST method

- *main.py* (scratch) code to make a POST request

- *post_request.py* (unfinished) code that will replace *main.py*, ties all the code together into 1 module

In the *src/* directory, you'll find:

- *preprocess.py* which encodes url parameters for sending a request to Google Maps api

- *client.py* sends a request to Google Maps api, returns response as a Python bytes object

- *model_utils.py* helper module for running predictions through the YOLOv5 model, using files sent as a post request to the flask api

- *postprocess.py* maps machine learning model output (response) back to each address's index

In *static/* directory, you'll find:

- *address_data.csv*, a copy of the google sheet containing addresses with relevant columns only

- *best.pt*, YOLOv5 model weights

- *data_index.pkl*, indicies of addresses used to map model output back to original address

- *img_bytes_list.pkl*, contains a Python list object of 10 images retrieved from the Google Maps api, in bytes, from a sample of 10 addresses
