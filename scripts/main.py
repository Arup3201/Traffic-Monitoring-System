import sys
import os
import cv2 as cv
import keras.backend as K
from yolov1.inference import detection
from yolov1.settings import BASE_PATH

def main():
    img_path = sys.argv[1]

    img = cv.imread(os.path.join(BASE_PATH, img_path))
    xy_min, xy_max, labels = detection(img)

    print(xy_min, xy_max, labels)

if __name__=="__main__":
    main()