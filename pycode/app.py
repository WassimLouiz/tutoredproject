from fastapi import FastAPI 
import uvicorn
import pickle
from pycode.models import Image
import easyocr
from PIL import Image as im
import matplotlib.pyplot  as plt
import re
import numpy as np
import cv2


app = FastAPI()

loaded_model = pickle.load( open( "model_test.h5", "rb" ) )

@app.get("/Mr.Mrs{name}")
def yolo(name):
    return {"hello {} and welcome to this API".format(name)}

@app.get("/")
def greet():
    return{"Hello"}

@app.post("/predict")
def prediction(req:Image):
    image = req.ImageString
    img_ori = plt.imread(image)
    img= cv2.resize(img_ori, (224,224))
    y=[]
    y.append(img)
    img_norm=np.array(y)/255
    predict = loaded_model.predict(img_norm)
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

    return{"ans":"{}".format(ch1)}
    
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
    return(num[0]+di+num[1])

if __name__=="__main__":
    uvicorn.run(app)
