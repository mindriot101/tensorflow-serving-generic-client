#!/usr/bin/env python


import cv2
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.predict_pb2 import PredictRequest
import grpc
import time


image_data = cv2.imread("image.jpeg")
inputs = np.array([image_data])

channel = grpc.insecure_channel("localhost:8900")
stub = PredictionServiceStub(channel)

request = PredictRequest()
request.model_spec.name = "retinanet"

request.inputs["inputs"].CopyFrom(
    tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape)
)

result = stub.Predict(request, 60)

boxes = tf.make_ndarray(result.outputs["detection_boxes"])
scores = tf.make_ndarray(result.outputs["detection_scores"])
labels = tf.make_ndarray(result.outputs["detection_classes"])
num_detections = tf.make_ndarray(result.outputs["num_detections"])


def box_normal_to_pixel(box, dim, scalefactor=1):
    height, width = dim[0], dim[1]
    ymin = int(box[0] * height * scalefactor)
    xmin = int(box[1] * width * scalefactor)
    ymax = int(box[2] * height * scalefactor)
    xmax = int(box[3] * width * scalefactor)
    return np.array([xmin, ymin, xmax, ymax])


for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if score < 0.3:
        break

    dim = image_data.shape
    box = box_normal_to_pixel(box, dim)
    b = box.astype(int)

    cv2.rectangle(image_data, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)

    cv2.imwrite("out.png", image_data)
