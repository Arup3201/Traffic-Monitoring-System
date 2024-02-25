import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import keras.backend as K
from keras.regularizers import l2
from .settings import S, B, C, IMG_SIZE


class Yolo_Reshape(keras.layers.Layer):
  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    idx1 = S * S * B
    idx2 = idx1 + S * S * C

    # class probabilities
    confs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S, S, B]))
    confs = K.sigmoid(confs)

    #confidence
    class_probs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S, S, C]))
    class_probs = K.softmax(class_probs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S, S, B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([boxes, confs, class_probs])
    return outputs
  

def yolov1():
    pretrained_model = keras.applications.VGG19(include_top=False, 
                                                weights="imagenet", 
                                                input_shape=[IMG_SIZE[0], IMG_SIZE[1], 3])
    
    for layer in pretrained_model.layers:
        layer.trainable = False

    last_layer = pretrained_model.get_layer('block5_pool')
    last_output = last_layer.output
    
    x = Flatten()(last_output)
    x = Dense(512)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(S*S*(B*5+C), activation="sigmoid")(x)
    x = Yolo_Reshape(target_shape=(S, S, B*5+C))(x)

    model = keras.models.Model(pretrained_model.input, x)
    return model