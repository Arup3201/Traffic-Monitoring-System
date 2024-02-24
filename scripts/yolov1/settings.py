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
CLASS2ID = {'car': 0, 'bus': 1, 'truck': 2, 'motorbike': 3, 'pedestrian': 4}
ID2CLASS = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorbike', 4: 'pedestrian'}
COLOR_TABLE = {'car': '#f03e3e', 'bus': '#ae3ec9', 'truck': '#7048e8', 'motorbike': '#1c7ed6', 'pedestrian': '#0ca678'}
THRESHOLD = 0.7
BOX_THICKNESS = 6