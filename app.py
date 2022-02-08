from asyncio.windows_events import NULL
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel
import pandas as pd
import joblib
import datetime
import json
# from fastapi.middleware.cors import CORSMiddleware
from flask import Flask,request,abort
app = Flask(__name__)
# app = FastAPI()
# origins = [
#     # "http://localhost:3000",
#     # "http://localhost:8000",
#     "*"
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

#Load Model, 
def loadConfig():
    global model,scaler,df,shift_data
    model = load_model("./assets/model.h5")
    scaler = joblib.load("./assets/scaler.save") 
    df = pd.read_csv("./assets/covid19_phuket.csv")
    shift_data = 7

#Preprocessing data
async def preprocess_data(df):
    #Transform data
    data=scaler.transform(df.reshape(-1,1))
    data_scaled = data[-shift_data:]
    return data_scaled

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


async def predict(data, next):
    prediction_list = data[-shift_data:]
    index = []
    dates = []
    date = datetime.datetime.strptime(df["txn_date"].values[-1] , '%Y-%m-%d').date()
    for i in range(next):
        x = prediction_list[-shift_data:]
        x = x.reshape((1, shift_data, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
        dates.append(date + datetime.timedelta(days=i+1))
        index.append(i)
    prediction_list = prediction_list[shift_data-1:]
    prediction_list =scaler.inverse_transform(np.array(prediction_list).reshape(-1,1)).reshape(-1).tolist()
    print(prediction_list)
    return index,dates,prediction_list


loadConfig()

# data = requests.get("https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-by-provinces")
# df = pd.read_json(data.text)

# @app.get('/raw/')
# @app.route('/raw/')
async def get_raw():
    # date = datetime.datetime.strptime(df["txn_date"][0].values , '%Y-%m-%d').date()
    date= df["txn_date"].values
    # print(date)
    data = df["new_case"].values
    date_ = []
    data_ = []
    for i in range(len(date)):
        date_.append(str(date[i]))
        data_.append(int(data[i]))
    # print(data)
    res = {"date":date_,"data":data_}
    return res

# @app.get('/predict/')
@app.route('/predict/')
async def get_predict():
    df_pre = await preprocess_data(df["new_case"].values)
    y = await predict(df_pre,7)
    index ,date, data = y
    # print(index)
    # print(date)
    # print(data)
    

    
    pred = {"index":index,"date":date,"data":data}
    
    raw = await get_raw()
    
    return {
        "totalCase": 1,
        "todayCase": 2,
        "PredictTommorrow": 3,
        "allData":{"raw":raw,"predict":pred}}

    # return {"raw":raw,"predict":pred}

if __name__ == "__main__":
    # app.run(debug=False)
    app.run(host='0.0.0.0',port=8080)