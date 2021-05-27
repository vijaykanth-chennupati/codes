import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump

set = pd.read_csv('./SalaryData.csv')
x = set['YearsExperience']
y = set['Salary']
x = x.values
x = x.reshape(-1,1)
y = y.values
y = y.reshape(-1,1)
model = LinearRegression().fit(x,y)
dump(model,'trained_model.gz',compress=0)
x = float(input("Enter the experience: "))
output = model.predict([[x]])
output = str(int(output[0][0]))
print("The salary might be: "+output)

