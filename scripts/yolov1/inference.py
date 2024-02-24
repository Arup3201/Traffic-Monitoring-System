from .settings import *
from .model import yolov1
from .utils import yolo_head, xywh2minmax
import os
import cv2 as cv
import keras.backend as K

def detection(img, threshold=0.70):
    # img = cv.imdecode(img, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.
    img = K.expand_dims(img, axis=0) # ? * 448 * 448 * 3

    model = yolov1()
    model.load_weights(os.path.join(BASE_PATH, MODEL_DIR, MODEL_WEIGHTS))

    prediction = model.predict(img) # ? * 7 * 7 * 15

    predict_box = prediction[..., :B*4]  # ? * 7 * 7 * 8
    predict_trust = prediction[..., B*4:B*4+2]  # ? * 7 * 7 * 2
    predict_class = prediction[..., B*4+2:B*4+2+C]  # ? * 7 * 7 * 5

    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    predict_xy, predict_wh = yolo_head(_predict_box) # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.cast(predict_trust>=threshold, K.dtype(predict_xy)) # ? * 7 * 7 * 2
    box_mask = K.expand_dims(box_mask, axis=4) # ? * 7 * 7 * 2 * 1
    predict_xy = K.max(box_mask * predict_xy, axis=4) # ? * 7 * 7 * 2
    predict_wh = K.max(box_mask * predict_wh, axis=4) # ? * 7 * 7 * 2
    predict_class = K.argmax(predict_class, axis=-1) # ? * 7 * 7
    predict_class = K.expand_dims(predict_class, axis=3) # ? * 7 * 7 * 1

    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh) # ? * 7 * 7 * 2, ? * 7 * 7 * 2

    return predict_xy_min, predict_xy_max, predict_class # ? * 7 * 7 * 2, ? * 7 * 7 * 2, ? * 7 * 7 * 1