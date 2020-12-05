# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def quick_sort(arr):
    if type(arr) is not list:
        ls = np.array(arr).tolist()
    else:
        ls = arr

    if len(ls) < 2:
        return ls
    else:
        pivot = ls[0]
        less = [_ for _ in arr if _ <= pivot]
        greater = [_ for _ in arr if _ > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)


def nms(bounding_boxes, confidence_score, threshold):
    """
    Non-max Suppression Algorithm

    @param bounding_boxes list  Object candidate bounding boxes
    @param confidence_score list  Confidence score of bounding boxes
    @param threshold float IoU threshold

    @return Rest boxes after nms operation
    """

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_index = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_index.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_index, picked_boxes, picked_score
