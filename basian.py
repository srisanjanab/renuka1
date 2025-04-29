import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetWork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableEstimator
data = ('age'=[]
        'chol'=[]
        'fbs'=[]
        'restecg'=[]
        'thalach'=[]
        'target'=[]
        )
heartDisease=pd.DataFrame(data)
heartDisease.to_csv('heartdisease_csv',index=False)

model = DiscreteBayesianModel([
    ('age', 'fbs'), 
    ('fbs', 'target'), 
    ('target', 'restecg'), 
    ('target', 'thalach'), 
    ('target', 'chol')
])
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['target'], evidence={'age': 37})         
print(q)
