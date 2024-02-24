from .settings import *
from .model import yolov1
from .utils import yolo_head, xywh2minmax
import os
import cv2 as cv
import keras.backend as K
import tensorflow as tf

def detect(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.
    img = K.expand_dims(img, axis=0) # ? * 448 * 448 * 3

    model = yolov1()
    model.load_weights(os.path.join(BASE_PATH, MODEL_DIR, MODEL_WEIGHTS))

    prediction = model.predict(img) # ? * 7 * 7 * 15

    predict_box = prediction[..., :B*4]  # ? * 7 * 7 * 8
    predict_score = prediction[..., B*4:B*4+2]  # ? * 7 * 7 * 2
    predict_class = prediction[..., B*4+2:B*4+2+C]  # ? * 7 * 7 * 5

    predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    predict_xy, predict_wh = yolo_head(predict_box) # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    
    predict_class = K.expand_dims(predict_class, axis=3)

    return predict_xy, predict_wh, predict_score, predict_class # ? * 7 * 7 * 2 * 4, ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2, ? * 7 * 7 * 5