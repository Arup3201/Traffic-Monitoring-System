import os
from .utils import read_json
from .settings import TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, BATCH_SIZE, DATA_DIR
from .data import YoloDataset
from .model import yolov1, pretrained_yolov1
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from .settings import MODEL_DIR, MODEL_WEIGHTS, EPOCHS
from .loss import yolo_loss

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr

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