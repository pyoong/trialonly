# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:51:16 2021

@author: HP
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd

#app = Flask(__name__)
app = Flask(__name__, template_folder='template')

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        #Salary = int(request.form['Year'])
        Salary = int(request.form['Salary'])
        LastSalaryHike = int(request.form['LastSalaryHike'])
        TotalWorkExperience = float(request.form['TotalWorkExperience'])   
        AdaptabilityandInitiative = int(request.form['AdaptabilityandInitiative'])
        AttendanceandPunctuality = int(request.form['AttendanceandPunctuality'])
        LeadershipSkills = int(request.form['LeadershipSkills'])
        QualityofDeliverables = int(request.form['QualityofDeliverables'])
        JudgementandDecisionMaking = int(request.form['JudgementandDecisionMaking'])
        TeamWorkSkills = int(request.form['TeamWorkSkills'])
        CustomerRelationSkills = int(request.form['CustomerRelationSkills'])
        JobKnowledge = int(request.form['JobKnowledge'])
        ReliabilityandDependability = int(request.form['ReliabilityandDependability'])
                 
        # Prediction
        #input = [[Salary,LastSalaryHike,TotalWorkExperience,AdaptabilityandInitiative,AttendanceandPunctuality,LeadershipSkills,QualityofDeliverables,JudgementandDecisionMaking,TeamWorkSkills,CustomerRelationSkills,JobKnowledge,ReliabilityandDependability]]
        #input = [79000,3,2,5,5,3,5,5,2,2,3,3]  # to test
        #process_input = [np.array(input)]
        #normalize = scaler.transform(process_input)
        #prediction = model.predict(normalize)
        
        # Prediction
        #input = [int(x) for x in request.form.values()]
        #final_features = [np.array(input)]
        #norm = scaler.transform(final_features)  # to normalize the input values
        #prediction = model.predict(norm)
        
        
        new_data = [[Salary,LastSalaryHike,TotalWorkExperience,AdaptabilityandInitiative,AttendanceandPunctuality,LeadershipSkills,QualityofDeliverables,JudgementandDecisionMaking,TeamWorkSkills,CustomerRelationSkills,JobKnowledge,ReliabilityandDependability]]
        new_data1 = np.array(new_data).reshape(1,12)
        new_data_scaled = scaler.transform(new_data1)
        prediction = model.predict(new_data_scaled)
        
        
        #prediction = model.predict([[Salary,LastSalaryHike,TotalWorkExperience,AdaptabilityandInitiative,AttendanceandPunctuality,LeadershipSkills,QualityofDeliverables,JudgementandDecisionMaking,TeamWorkSkills,CustomerRelationSkills,JobKnowledge,ReliabilityandDependability]])        
        
        if prediction == 2:
            return render_template('index.html', prediction_texts= ('Predicted Result:', str(prediction) + 'Predicted to leave the company'))
        else:
            return render_template('index.html', prediction_texts= ('Predicted Result:', str(prediction) + 'Expected to remain in the company'))

    else:
        return render_template('index.html')
#    )
if __name__=="__main__":
    app.run(debug=True)