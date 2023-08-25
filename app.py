from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Time=request.form.get('Time'),
            V1=request.form.get('V1'),
            V2=request.form.get('V2'),
            V3=request.form.get('V3'),
            V4=request.form.get('V4'),
            V5=request.form.get('V5'),
            V6=request.form.get('V6'),
            V7=request.form.get('V7'),
            V8=request.form.get('V8'),
            V9=request.form.get('V9'),
            V10=request.form.get('V10'),
            V11=request.form.get('V11'),
            V12=request.form.get('V12'),
            V13=request.form.get('V13'),
            V14=request.form.get('V14'),
            V15=request.form.get('V15'),
            V16=request.form.get('V16'),
            V17=request.form.get('V17'),
            V18=request.form.get('V18'),
            V19=request.form.get('V19'),
            V20=request.form.get('V20'),
            V21=request.form.get('V21'),
            V22=request.form.get('V22'),
            V23=request.form.get('V23'),
            V24=request.form.get('V24'),
            V25=request.form.get('V25'),
            V26=request.form.get('V26'),
            V27=request.form.get('V27'),
            V28=request.form.get('V28'),
            Amount=request.form.get('Amount'),
        )

        print("yes")
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True)        


