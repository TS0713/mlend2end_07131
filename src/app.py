from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.pipeline.train_pipeline import train_the_model
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/modeltraining",methods=["GET","POST"])
def train_model():
    train_the_model()
    return "Model Training Completed"
    
@app.route("/prediction",methods=["GET","POST"])
def predict_data():

    try:

        if request.method=="GET":
            return render_template("home.html")
        else:
            prediction_obj = PredictPipeline()
            data_obj = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
                )
            user_processed_data = data_obj.feature_processing()
            prediction_obj = PredictPipeline()
            results=prediction_obj.predict(user_processed_data)
            return render_template("home.html",results=results[0])
    except Exception as e:
        logging.info(CustomException(str(e),sys))
        raise CustomException(str(e),sys)



if __name__=="__main__":
    app.run(host="0.0.0.0",port=8002,debug=True)
