# -*- coding: utf-8 -*-

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

loaded_model=pickle.load(open("C:/projectML/trainedModel.sav",'rb'))
scaler=pickle.load(open("scaler.pkl",'rb'))

input_data=(3,78,50,32,88,31,0.248,26)
input_predict_data=np.asarray(input_data)
input_predict_data=input_predict_data.reshape(1,-1)
std_data=scaler.transform(input_predict_data)
prediction=loaded_model.predict(input_predict_data)
if prediction[0]==0:
  print("Congratulations! you are not diabetic")
else:
  print("Hey! You may have debetes. Please consult a doctor")
