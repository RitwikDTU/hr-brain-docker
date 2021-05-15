from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import logging
from imblearn.over_sampling import SMOTE

app = Flask(__name__, template_folder="template")
model = pickle.load(open("./models/lgbm.pkl", "rb"))
print("Model Loaded")


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        # city
        city = float(request.form['city'])
        # city development index
        city_development_index = float(request.form['city_development_index'])
        # gender
        gender = float(request.form['gender'])
        # relevant_experience
        relevant_experience = float(request.form['relevant_experience'])
        # enrolled_university
        enrolled_university = float(request.form['enrolled_university'])
        # education_level
        education_level = float(request.form['education_level'])
        # major_discipline
        major_discipline = float(request.form['major_discipline'])
        # experience
        experience = float(request.form['experience'])
        # company size
        company_size = float(request.form['company_size'])
        # company_type
        company_type = float(request.form['company_type'])
        # lastnewjob
        lastnewjob = float(request.form['lastnewjob'])
        # training_hours
        training_hours = float(request.form['training_hours'])

        input_lst = [[city, city_development_index, gender, relevant_experience, enrolled_university, education_level,
                     major_discipline, experience, company_size, company_type, lastnewjob, training_hours]]
   
        pred = model.predict(input_lst)
        output = pred
        if output == 0:
            return render_template("predict_0.html")
        else:
            return render_template("predict_1.html")
    return render_template("predictor.html")


@app.route("/monitor_", methods=['GET', 'POST'])
@cross_origin()
def monitor_():
    if request.method == "POST":
        # boosting_type
        boosting_type = int(request.form['boosting_type'])
        if boosting_type==0:
            boosting_type='gbdt'
        if boosting_type==1:
            boosting_type='dart'
        if boosting_type==2:
            boosting_type='goss'
        if boosting_type==3:
            boosting_type='rf'
        # learning_rate
        learning_rate = float(request.form['learning_rate'])
        # n_estimators
        n_estimators = int(request.form['n_estimators'])
        # max_depth
        max_depth = int(request.form['max_depth'])
        # num_leaves
        num_leaves = int(request.form['num_leaves'])
        # min_child_samples
        min_child_samples = int(request.form['min_child_samples'])
        # n_jobs
        n_jobs = int(request.form['n_jobs'])
        # bagging_freq
        bagging_freq = int(request.form['bagging_freq'])
        # bagging_fraction
        bagging_fraction = float(request.form['bagging_fraction'])

   
        # default values boosting_type='gbdt', learning_rate=0.1, n_estimators=100,
        # max_depth=-1, num_leaves=31, min_child_samples=20, n_jobs=-1,
        # bagging_freq=0, bagging_fraction=0.9
        # gbdt=0,rf=3,dart=1,goss=2
        
        
        GOOGLE_APPLICATION_CREDENTIALS='hranalytics-key.json'
        MLFLOW_TRACKING_USERNAME='ritwikdtu'
        MLFLOW_TRACKING_PASSWORD='ritwik2392'

        os.environ['GOOGLE_APPLICATION_CREDENTIALS']=GOOGLE_APPLICATION_CREDENTIALS
        os.environ['MLFLOW_TRACKING_USERNAME']=MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD']=MLFLOW_TRACKING_PASSWORD
        
        
        
        
        np.random.seed(40)

        # Read the csv file
        df=pd.read_csv("processed_df.csv")
        df.drop('Unnamed: 0', axis=1, inplace=True)
        X=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        # Split the data into training and test sets. (0.75, 0.25) split.
        X, y = SMOTE().fit_resample(X, y)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.25, random_state = 0)
        scaler = StandardScaler()
        train_x.iloc[:,[1,7,8,10,11]]=scaler.fit_transform(train_x.iloc[:,[1,7,8,10,11]])
        test_x.iloc[:,[1,7,8,10,11]]=scaler.transform(test_x.iloc[:,[1,7,8,10,11]])
        
        
        #experiment_name="hrAnalytics"
        
        #if not mlflow.get_experiment_by_name(experiment_name):
            #mlflow.create_experiment(name=experiment_name)
            
        #mlflow.set_experiment(experiment_name)
        #experiment=mlflow.get_experiment_by_name(experiment_name)
        
        
        #experiment_name='Experiment 1'
        tracking_uri='http://35.222.184.89' # external IP
        
        #mlflow.set_experiment(experiment_name)
        #experiment=mlflow.get_experiment_by_name(experiment_name)
        
        
        mlflow.set_tracking_uri(tracking_uri)
        
        with mlflow.start_run():
            lgbm = lgb.LGBMClassifier(boosting_type=boosting_type, learning_rate=learning_rate, n_estimators=n_estimators,
                                    max_depth=max_depth, num_leaves=num_leaves, min_child_samples=min_child_samples, n_jobs=n_jobs,
                                    bagging_freq=bagging_freq, bagging_fraction=bagging_fraction)
            lgbm.fit(train_x, train_y)

            predicted_qualities = lgbm.predict(test_x)

            
            rmse=np.sqrt(mean_squared_error(test_y, predicted_qualities))
            mae = mean_absolute_error(test_y, predicted_qualities)
            r2 = r2_score(test_y, predicted_qualities)
            roc = roc_auc_score(test_y, predicted_qualities)
            class_report = classification_report(predicted_qualities, test_y)
         

            #print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            #print("  RMSE: %s" % rmse)
            #print("  MAE: %s" % mae)
            #print("  R2: %s" % r2)

            mlflow.log_param("boosting_type", boosting_type)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("num_leaves", num_leaves)
            mlflow.log_param("min_child_samples", min_child_samples)
            mlflow.log_param("n_jobs", n_jobs)
            mlflow.log_param("bagging_freq", bagging_freq)
            mlflow.log_param("bagging_fraction", bagging_fraction)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("roc_score", roc)
            
            
            

            #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            #if tracking_url_type_store != "file":
                #Register the model
                #mlflow.sklearn.log_model(lgbm, "model", registered_model_name="LightBGM")
            #else:
            #mlflow.sklearn.log_model(lgbm, "model")
            
            return render_template("modelMonitor.html")
        
    
    return render_template("monitor.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
