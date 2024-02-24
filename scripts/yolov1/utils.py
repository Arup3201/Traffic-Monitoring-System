import numpy as np
import cv2 as cv
import joblib
import keras.backend as K
from .settings import *

def hex_to_bgr(hex_color):
  """Converts a hexadecimal color code to a BGR tuple.

  Args:
    hex_color: A hexadecimal color code, such as '#FFFFFF'.

  Returns:
    A tuple of three integers representing the BGR values of the color.
  """

  hex_color = hex_color.lstrip('#')
  r = int(hex_color[0:2], 16)
  g = int(hex_color[2:4], 16)
  b = int(hex_color[4:6], 16)
  return (b, g, r)

def read_json(filename):
    json = joblib.load(filename)
    return json

def read_single_example(image_path, bboxes, label):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_h, image_w = image.shape[0:2]
    image = cv.resize(image, (448, 448))
    image = image / 255.

    label_matrix = np.zeros([S, S, B*5+C])
    for bbox in bboxes:
        x, y, w, h = bbox

        x = x / image_w
        y = y / image_h
        w = w / image_w
        h = h / image_h

        loc = [S * x, S * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j

        if label_matrix[loc_i, loc_j, 4] == 0:
            label_matrix[loc_i, loc_j, B*5+label] = 1
            label_matrix[loc_i, loc_j, :4] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 4] = 1  # response

    return image, label_matrix

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * IMG_SIZE[0]
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh

def get_detection_data(xy, wh, scores, labels):
    detections = {'bboxes': [], 'classes': [], 'scores': []}

    xy, wh, scores, labels = xy[0, ...], wh[0, ...], scores[0, ...], labels[0, ...]

    for box_id in range(B):
        for r in range(S):
            for c in range(S):
                if(scores[r, c, box_id] >= THRESHOLD):
                    x, y, w, h = xy[r, c, box_id, 0].numpy(), xy[r, c, box_id, 1].numpy(), wh[r, c, box_id, 0].numpy(), wh[r, c, box_id, 1].numpy()
                    detections['bboxes'].append([x, y, w, h])
                    label = np.argmax(labels[r, c, :])
                    detections['classes'].append(ID2CLASS[label])
                    detections['scores'].append(scores[r, c, box_id])

    return detections

def draw_boxes(img, detections):
    scale = max(img.shape[0:2]) / IMG_SIZE[0]

    for bbox, label, score in zip(detections['bboxes'], detections['classes'], detections['scores']):
        x, y, w, h = bbox
        x1 = int(x - w//2)
        y1 = int(y - h//2)
        x2 = int(x + w//2)
        y2 = int(y + h//2)
        cv.rectangle(img, (x1, y1), (x2, y2), color=hex_to_bgr(COLOR_TABLE[label]), thickness=BOX_THICKNESS)

        text = f'{label} {score:.2f}'
        font = cv.FONT_HERSHEY_DUPLEX
        font_scale = max(0.3*scale, 0.3)
        thickness = max(int(1 * scale), 1)
        cv.putText(img, text, (x1, y1), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    return img