from fastapi import FastAPI, File, UploadFile, HTTPException
import pickle
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # only errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#from pydantic import BaseModel

app = FastAPI(
    title="Api for image classification",
    summary="Takes in Image and output the name of the image taken In",
    version="1.0.0"
)
class_name = [
 'Plane' , 'Car' , 'Bird' , 'Cat' , 'Deer' ,'Dog' , 'Frog' , 'Horse' , 'Ship' , 'Truck'
]


with open('model_best1.pickle', 'rb') as f:
    model = pickle.load(f)

warmed_up = False

@app.get("/")
#async 
def root():
    global warmed_up
    if not warmed_up:
        dummy_img = cv.imread('car_32x23.jpeg')
        dummy_img = np.expand_dims(dummy_img, 0)
        _ = model.predict((np.array(dummy_img)/255), verbose=0)
        warmed_up = True
    return {"message": "Model ready!"}
@app.post("/predict")
async def predict_object(image: UploadFile=File(...)):
    #check file type
    if image.content_type not in ('image/jpeg', 'image/png', 'image/jpg'):
        raise HTTPException(status_code=400, detail="Invalid Image type")

    #converting file into byte
    file_byte = await image.read()

    #convert byte into 1D array
    np_arr = np.frombuffer(file_byte, np.uint8)
    #decode image as BGR
    img_bgr = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    content = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    content = content/ 255
    def resize(img):
        new_size = (32,32)
        img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
        return img
    img = resize(content)
    img = np.expand_dims(img, axis=0)  
    prediction = model.predict(img)
    index = np.argmax(prediction)
    predicted_class = f"The model predicts {class_name[index]} with a probability of approximately {np.max(prediction) *100 :.2f}%"
    return predicted_class

