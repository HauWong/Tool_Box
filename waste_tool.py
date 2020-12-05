# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
from osgeo import gdal

import roi_extract as re


def paint_rectangles(image, box_ls, color=(255, 0, 0), width=2):

    """
    在图像上批量绘制矩形框

    Args：
        image: array, 图像数组, shape=(w, h, c)
        box_ls: list, 矩形框列表, [(row_min, row_max, col_min, col_max), ...]
        color: tuple, 8-bit RGB形式的框体颜色
        width: int, 框线宽度
    Returns：
        image: array, 结果图像
    """
    img_w, img_h, _ = image.shape
    boundary = (0, img_h, 0, img_w)
    for box in box_ls:
        box = re.limit_boundary(box, boundary)
        first_point = (box[2], box[0])
        last_point = (box[3], box[1])
        cv2.rectangle(image, first_point, last_point, color, width)
    return image


def paint_ssd_rectangles(image, box_ls, color=(0, 0, 255), width=2, show_conf=True):

    """
    在图像上批量绘制矩形框

    Args：
        image: array, 图像数组, shape=(w, h, c)
        box_ls: list, 矩形框列表, [(prob, xmin, ymin, xmax, ymax), ...]
        color: tuple, 8-bit RGB形式的框体颜色
        width: int, 框线宽度
    Returns：
        image: array, 结果图像
    """
    img_w, img_h, _ = image.shape
    boundary = (0, img_h, 0, img_w)
    for box in box_ls:
        box = re.limit_boundary(box, boundary)
        first_point = (int(box[1]), int(box[2]))  # 正确方式
        last_point = (int(box[3]), int(box[4]))
        if show_conf:
            cv2.putText(image, '%.3f'%box[0], first_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, width)
        cv2.rectangle(image, first_point, last_point, color, width)
    return image


def read_label(label_path):
    """
    Read label file(*.tif)
    :param label_path: str, label path
    :return: array of the label above
    """
    gdal.AllRegister()
    lab_dataset = gdal.Open(label_path)
    return lab_dataset.ReadAsArray()


def mask_roi(mask_data):
    """
    Get rectangles from mask data
    :param mask_data: array, shape=(height, width)
    :return: raster num, image rows, image cols, rectangles list
    """
    rows, cols = mask_data.shape
    rois_num = mask_data.max()
    res = []
    for i in range(1, rois_num+1):
        mask_idxs = np.argwhere(mask_data == i)
        if mask_idxs.shape[0] == 0:
            continue
        rect = re.limit_boundary(re.get_extent(mask_idxs), (0, rows, 0, cols))
        res.append(rect)
    return res


def box_5_rect(box, rect_size):
    """
    Generate 5 rectangles around specific box
    ps. add a random bias
    :param box: list or tuple, box boundary, (xmin, ymin, xmax, ymax)
    :param rect_size: int, size of the rectangle
    :return: list, rectangles, [left-up, right-up, right-bottom, left-bottom, center]
    """
    res = []
    rnd = random.randint(10, 20)
    box = [box[0]-rnd, box[1]-rnd, box[2]+rnd, box[3]+rnd]
    res.append([box[1], box[0], box[1]+rect_size, box[0]+rect_size])
    res.append([box[1], box[2]-rect_size, box[1]+rect_size, box[2]])
    res.append([box[3]-rect_size, box[2]-rect_size, box[3], box[2]])
    res.append([box[3]-rect_size, box[0], box[3], box[0]+rect_size])
    center_y = int((box[3]+box[1])/2)
    center_x = int((box[2]+box[0])/2)
    diff = int(rect_size/2)
    res.append([center_y-diff, center_x-diff, center_y+diff, center_x+diff])

    return res


def is_unqualified(box_list, threshold):
    """
    Judge if the size of any box satisfies the condition,
    if anyone is less than the threshold, return True, or return False.
    :param box_list: list or tuple, [(row_min, row_max, col_min, col_max), ...]
    :param threshold: float
    :return: Boolean
    """
    for b in box_list:
        if b[1]-b[0] < threshold or b[3]-b[2] < threshold:
            return True
    return False


def extract_box(box_str):
    """
    Extract box from the string of detection box
    :param box_str: str, detection box, "0 0.98 10 20 150 165"
    :return: dict
    """
    eles = box_str.split(' ')
    return {'file': eles[0],
            'box': [float(_) for _ in eles[1:]]}


def update_ssd_box(origin_coord, box):
    """
    Update the boundary of box from ssd detection
    :param origin_coord: coordinate of the the origin point
    :param box: list or tuple, conf and boundary of box, (conf, xmin, ymin, xmax, ymax)
    :return: list, new box
    """
    y_o, x_o = origin_coord
    return [box[0], box[1]+x_o, box[2]+y_o, box[3]+x_o, box[4]+y_o]


def update_yolo_box(origin_coord, box):
    """
    Update the boundary of box from yolo detection
    :param origin_coord: coordinate of the the origin point
    :param box: list or tuple, conf and boundary of box, (x_center, y_center, width, height, conf)
    :return: list, new box
    """
    y_o, x_o = origin_coord
    x_c, y_c, w, h, conf = box
    x_min = x_c - w/2 + x_o
    y_min = y_c - h/2 + y_o
    x_max = x_c + w/2 + x_o
    y_max = y_c + h/2 + y_o
    return [conf, x_min, y_min, x_max, y_max]


def box_merge(box_list, confidences, threshold):

    """
    Merge boxes
    :param box_list: list, boxes, [(xmin, ymin, xmax, ymax)]
    :param confidences: list, confidences of all boxes
    :param threshold: float, iou threshold
    :return: list, all boxes merged
    """

    def intersection(b_ls):

        b_ls = np.array(b_ls)
        x0 = np.max(b_ls[:, 0])
        y0 = np.max(b_ls[:, 1])
        x1 = np.min(b_ls[:, 2])
        y1 = np.min(b_ls[:, 3])
        return x0, y0, x1, y1

    def union(b_ls):
        b_ls = np.array(b_ls)
        x0 = np.min(b_ls[:, 0])
        y0 = np.min(b_ls[:, 1])
        x1 = np.max(b_ls[:, 2])
        y1 = np.max(b_ls[:, 3])
        return x0, y0, x1, y1

    def enough_iou(b_ls, thr):

        inter = intersection(b_ls)
        w = np.maximum(0.0, inter[2] - inter[0])
        h = np.maximum(0.0, inter[3] - inter[1])
        area_inter = w * h

        # union
        un = union(b_ls)
        w = np.maximum(0.0, un[2] - un[0])
        h = np.maximum(0.0, un[3] - un[1])
        area_un = w * h

        iou = area_inter/area_un
        if iou > thr:
            return True
        else:
            return False

    def in_another(b1, b2, thr):

        w = np.maximum(0.0, b1[2] - b1[0])
        h = np.maximum(0.0, b1[3] - b1[1])
        area_b1 = w * h

        w = np.maximum(0.0, b2[2] - b2[0])
        h = np.maximum(0.0, b2[3] - b2[1])
        area_b2 = w * h

        inter = intersection([b1, b2])
        w = np.maximum(0.0, inter[2] - inter[0])
        h = np.maximum(0.0, inter[3] - inter[1])
        area_inter = w * h

        iob1 = area_inter/area_b1
        iob2 = area_inter/area_b2
        if iob1 > thr or iob2 > thr:
            return True
        else:
            return False

    if len(box_list) == 0:
        return []

    parts = []
    for b in box_list:
        print(b)
        if len(parts) == 0:
            parts.append([b])
        else:
            flag = False
            for sub_boxes in parts:
                if enough_iou(sub_boxes+[b], threshold) or in_another(b, union(sub_boxes), threshold):
                    sub_boxes.append(b)
                    flag = True
                    break
            if not flag:
                parts.append([b])

    return [union(_) for _ in parts]