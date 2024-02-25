from keras.callbacks import ModelCheckpoint
import os
from yolov1.settings import MODEL_DIR, MODEL_WEIGHTS, EPOCHS
from yolov1.loss import yolo_loss
from yolov1.utils import read_json
from yolov1.settings import TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, BATCH_SIZE, DATA_DIR
from yolov1.data import YoloDataset
from yolov1.model import yolov1, pretrained_yolov1
from yolov1.train import CustomLearningRateScheduler, lr_schedule

if __name__=="__main__":
    train_annotations = read_json(TRAIN_ANNOTATIONS)
    val_annotations = read_json(VAL_ANNOTATIONS)

    train_gen = YoloDataset(train_annotations, BATCH_SIZE, DATA_DIR)
    val_gen = YoloDataset(val_annotations, BATCH_SIZE, DATA_DIR)

    model = pretrained_yolov1()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    mcp_save = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_WEIGHTS),
                            save_best_only=True,
                            monitor='val_loss',
                            mode='min')

    model.compile(loss=yolo_loss, optimizer='adam')

    model.fit(
        x=train_gen,
        steps_per_epoch=len(train_annotations)//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_annotations)//BATCH_SIZE,
        callbacks=[
            CustomLearningRateScheduler(lr_schedule),
            mcp_save
        ]
    )