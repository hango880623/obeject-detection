# To reproduce the conda enviornment

**Note** the example commands used below are for Unix/Linux operating systems

``` conda env create -f environment.yml ```

The new conda enviornment will be named *yolo*.

To activate the enviorment run

``` conda activate yolo ```
or
``` source activate yolo ```

## To get the flask API running locally on your machine

First clone this repository

``` git clone https://github.com/soph-py/object_detection.git ```

Once the conda enviorment is activated, change directories to app/

``` cd ~/app ```

Then start running the Flask app (making sure you have the yolo enviorment activated and flask installed) by running the following:

``` python app.py ```

Once the Flask app is running, run the code in *main.py* to make a test post request
