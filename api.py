from asyncio.windows_events import NULL
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from fastapi import FastAPI
# from pydantic import BaseModel
import pandas as pd
import joblib
import datetime

app = FastAPI()

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
async def predict(data,next):
    y_preds = []
    dates = []
    index = []
    date = datetime.datetime.strptime(df["txn_date"].values[-1] , '%Y-%m-%d').date()
    for i in range(next):
        if i == 0:
            x_pred = data[-shift_data:]
        else:
            x_pred = np.append(x_pred[0][:-1],[[y_pred]],axis=0)
        x_pred = np.array([x_pred.tolist()])
        y_pred = model.predict(x_pred,batch_size=1)[0][0]
        y_preds.append(y_pred)
        dates.append(date + datetime.timedelta(days=i+1))
        index.append(i)
    y = scaler.inverse_transform(np.array(y_preds).reshape(-1,1)).reshape(-1).tolist()
    return index, dates, y



loadConfig()

# data = requests.get("https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-by-provinces")
# df = pd.read_json(data.text)

@app.get('/getclass/')
async def get_class():
    df_pre = await preprocess_data(df["new_case"].values)
    y = await predict(df_pre,5)
    index ,date, data = y
    res = {"index":index,"date":date,"data":data}
    return {"result":res}