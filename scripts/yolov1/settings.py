import os

S = 7
B = 2
C = 5
BATCH_SIZE = 4
IMG_SIZE = (448, 448)
DATA_DIR = '.data'
TRAIN_ANNOTATIONS = '.data/json_annotations/train_annotations.json'
VAL_ANNOTATIONS = '.data/json_annotations/val_annotations.json'
MODEL_DIR = 'model'
MODEL_WEIGHTS = 'yolov1.h5'
EPOCHS = 2
BASE_PATH = os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir)