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
    lrelu = keras.layers.LeakyReLU(alpha=0.1)

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size= (7, 7), strides=(1, 1), input_shape =(IMG_SIZE[0], IMG_SIZE[1], 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=192, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=128, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(S*S*(B*5+C), activation='sigmoid'))
    model.add(Yolo_Reshape(target_shape=(S, S, B*5+C)))

    return model

def pretrained_yolov1():
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