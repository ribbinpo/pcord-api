from asyncio.windows_events import NULL
from re import A
import numpy as np
from fastapi import FastAPI
# from pydantic import BaseModel
import pandas as pd
import joblib
import requests
import os
#For Train model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import math





app = FastAPI()

#Load Model, 
def loadConfig():
    global model,scaler,phuket,shift_data,batch_size
    data = requests.get("https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-by-provinces")
    df = pd.read_json(data.text)
    phuket = phuket_case(df)
    shift_data = 7
    batch_size = 1

# split Phuket
def phuket_case(df):
    phuketAll=df[df["province"] == "ภูเก็ต"].reset_index()
    phuket=phuketAll[phuketAll["new_case"] != 0]
    phuketBefore = phuketAll.loc[phuket.index[0]-1]
    #Reset value before
    phuket["total_case"]=phuket["total_case"] - phuketBefore["total_case"]
    phuket["total_case_excludeabroad"] -= phuketBefore["total_case_excludeabroad"]
    phuket["total_death"] -= phuketBefore["total_death"]
    df = phuket.reset_index()
    return df

# Function Create batch
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i+look_back])
	return np.array(dataX), np.array(dataY)

#Preprocessing data
async def preprocess_data(df):
    #Transform data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df.reshape(-1,1))
    return df,scaler

def get_model():
    #Create Model
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(6, batch_input_shape=(batch_size, shift_data, 1), stateful=True, return_sequences=True))
    model.add(LSTM(6, batch_input_shape=(batch_size, shift_data, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
#Predict data
# async def predict(data,next):
#     y_preds = []
#     dates = []
#     index = []
#     date = datetime.datetime.strptime(df["txn_date"].values[-1] , '%Y-%m-%d').date()
#     for i in range(next):
#         if i == 0:
#             x_pred = data[-shift_data:]
#         else:
#             x_pred = np.append(x_pred[0][:-1],[[y_pred]],axis=0)
#         x_pred = np.array([x_pred.tolist()])
#         y_pred = model.predict(x_pred,batch_size=1)[0][0]
#         y_preds.append(y_pred)
#         dates.append(date + datetime.timedelta(days=i+1))
#         index.append(i)
#     y = scaler.inverse_transform(np.array(y_preds).reshape(-1,1)).reshape(-1).tolist()
#     return index, dates, y

loadConfig()


@app.get('/train/')
async def get_train():
    
    df_pre,scaler = await preprocess_data(phuket["new_case"].values)
    trainX, trainY = create_dataset(df_pre, shift_data)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    model = get_model()
    # fit model
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    #Evaluation model

    #Save csv, model and Scaler
    path = "./assets/new/"
    model.save(os.path.join(path,"model.h5"))
    phuket.to_csv(os.path.join(path,"covid19_phuket.csv"))
    joblib.dump(scaler, os.path.join(path,"scaler.save")) 

    return "success"