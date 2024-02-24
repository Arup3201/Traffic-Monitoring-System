import sys
import os
import cv2 as cv
import keras.backend as K
from yolov1.inference import detect
from yolov1.utils import get_detection_data, draw_boxes, non_max_suppression
from yolov1.settings import BASE_PATH
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    img_path = sys.argv[1]

    img = cv.imread(os.path.join(BASE_PATH, img_path))
    xy, wh, scores, labels = detect(img)
    detection_data = get_detection_data(xy, wh, scores, labels)
    best_boxes, boxes = non_max_suppression(detection_data)
    img = draw_boxes(img, boxes, best_boxes, detection_data['scores'], detection_data['classes'])
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


if __name__=="__main__":
    tf.get_logger().setLevel('INFO')
    main()