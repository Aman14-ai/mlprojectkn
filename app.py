from flask import Flask , render_template, request
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging


app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        gender=request.form.get('gender')
        race_ethnicity=request.form.get('ethnicity')
        parental_level_of_education=request.form.get('parental_level_of_education')
        lunch=request.form.get('lunch')
        test_preparation_course=request.form.get('test_preparation_course')
        reading_score=float(request.form.get('writing_score'))
        writing_score=float(request.form.get('reading_score'))

        print(gender,type(gender))

        data=CustomData(
            gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score
        )
        df=data.get_data_as_data_frame()
        print(df)
        logging.info("going to predict the data")
        predict_pipeline=PredictPipeline()
        logging.info("mid predicting the data")
        print("Mid Prediction")
        results=predict_pipeline.predict(df)
        logging.info("after predicting the data")
        print("after Prediction")
        return render_template('home.html',results=np.round(results[0],2))
    
if __name__=="__main__":
    port = 5000
    print("running on http://localhost:",port)
    app.run(host="0.0.0.0",port=port)