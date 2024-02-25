import os
import numpy as np
import tensorflow as tf
from .utils import read_single_example


class YoloDataset(tf.keras.utils.Sequence) :
    def __init__(self, annotations, batch_size, data_dir) :
        self.annotations = annotations
        self.batch_size = batch_size
        self.data_dir = data_dir

    def __len__(self) :
        return (np.ceil(len(self.annotations) / float(self.batch_size))).astype(np.int32)


    def __getitem__(self, idx) :
        batch = self.annotations[idx * self.batch_size : (idx+1) * self.batch_size]

        train_image = []
        train_label = []

        for i in range(0, len(batch)):
            img_path = batch[i]['image_path']
            bboxes = batch[i]['bbox']
            label = batch[i]['class']
            image, label_matrix = read_single_example(os.path.join(self.data_dir, img_path), bboxes, label)
            train_image.append(image)
            train_label.append(label_matrix)

        return np.array(train_image), np.array(train_label)