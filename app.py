from fastapi import FastAPI 
import uvicorn
from tensorflow import keras
from models import Image
import easyocr
from PIL import Image as im
import re
import numpy as np
import pandas as pd
import cv2


app = FastAPI()

loaded_model = keras.models.load_model("model_test.h5")

@app.get("/{name}")
def yolo(name):
    return {"hello {} and welcome to this API".format(name)}

@app.get("/")
def greet():
    return{"Hello motherFucker"}

@app.post("/predict")
def dif(a, b):
    x=min(len(a),len(b))
    return ([i for i in range(x) if a[i] != b[i]])

def mod(ch):
    
    if (ch.find("-")>=0):
        return ch
    
    num = re.findall(r'\d+',ch)
    t=("").join(num)
    di=dif(t,num)
    num1=num
    num1.reverse()

    print(num[0])
    print(num[1])
    if len(di)>=2:
        di=" تونس "
    else:
        di=" ن ت "
    return(num1[0]+di+num1[1])

def prediction(req:Image):
    image = req.ImageString
    decoded_data = base64.b64decode(image)
    np_data = np.fromstring(decoded_data,np.uint8)
    img_ori = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    img= cv2.resize(img_ori, (224,224))
    y=[]
    y.append(img)
    img_norm=np.array(y)/255
    predict = model.predict(img_norm)
    ny = predict * 255
    image = cv2.rectangle(img,(int(ny[0][0])+30,int(ny[0][1])+30),(int(ny[0][2])-20,int(ny[0][3])),(0, 255, 0))


    cropped_image = image[int(ny[0][3]):int(ny[0][1])+20 , int(ny[0][2])-20:int(ny[0][0])+30]

    data = im.fromarray(cropped_image)
    data_ori = im.fromarray(img_ori)

    data = data.resize((450,150))
    numpydata = np.asarray(data)
    reader = easyocr.Reader(['ar'])
    result = reader.readtext(numpydata,x_ths=1,y_ths=1,text_threshold=0.6,paragraph=True)

    ch=result[0][1]
    ch1=mod(ch)

    return{"{}".format(ch1)}


if __name__=="__main__":
    uvicorn.run(app)
