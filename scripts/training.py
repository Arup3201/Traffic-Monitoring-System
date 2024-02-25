import os
import argparse
from yolov1.settings import *
from yolov1.loss import yolo_loss
from yolov1.utils import read_json
from yolov1.settings import *
from yolov1.data import YoloDataset
from yolov1.model import yolov1, pretrained_yolov1
from yolov1.train import CustomLearningRateScheduler, lr_schedule

if __name__=="__main__":
    parser = argparse.ArgumentParser('help in how to train the model and set parameters.')
    parser.add_argument('-e', '--epoch', help='set epochs for training the model')
    parser.add_argument('-b', '--batch', help='set the batch size')
    args = vars(parser.parse_args())
    
    epochs = int(args['epoch'])
    batch_size = int(args['batch'])

    train_annotations = read_json(TRAIN_ANNOTATIONS)
    val_annotations = read_json(VAL_ANNOTATIONS)

    train_gen = YoloDataset(train_annotations, batch_size, DATA_DIR)
    val_gen = YoloDataset(val_annotations, batch_size, DATA_DIR)

    model = pretrained_yolov1()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model.compile(loss=yolo_loss, optimizer='adam')

    model.fit(
        x=train_gen,
        steps_per_epoch=len(train_annotations)//batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=len(val_annotations)//batch_size,
        callbacks=[
            CustomLearningRateScheduler(lr_schedule)
        ]
    )

    model.save_weights(os.path.join(BASE_PATH, MODEL_DIR, MODEL_WEIGHTS))