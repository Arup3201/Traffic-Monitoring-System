from .settings import IMG_SIZE, MODEL_DIR, MODEL_WEIGHTS, B, C
from .model import yolov1
from .utils import yolo_head, xywh2minmax
import os
import cv2 as cv
import keras.backend as K

def detection(img_obj, threshold=0.70):
    image = cv.imdecode(img_obj, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, IMG_SIZE)
    image = image / 255.

    model = yolov1()
    model.load_weights(os.path.join(MODEL_DIR, MODEL_WEIGHTS))

    prediction = model.predict([image]) # ? * 7 * 7 * 15

    predict_box = prediction[..., :B*4]  # ? * 7 * 7 * 8
    predict_trust = prediction[..., B*4:B*4+2]  # ? * 7 * 7 * 2
    predict_class = prediction[..., B*4+2:B*4+2+C]  # ? * 7 * 7 * 5

    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    predict_xy, predict_wh = yolo_head(_predict_box) # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.cast(predict_trust>=threshold, K.dtype(predict_xy)) # ? * 7 * 7 * 2
    predict_xy = K.max(box_mask * predict_xy, axis=3)
    predict_wh = K.max(box_mask * predict_wh, axis=3)
    predict_class = K.argmax(predict_class, axis=-1)

    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)

    return predict_xy_min, predict_xy_max