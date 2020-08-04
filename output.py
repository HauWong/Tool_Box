# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import json
import numpy as np
import cv2
from osgeo import gdal

import transform
import basic_image_tool as bit


def extract_category(distribution, threshold=None, num=1):

    """ 提取类别索引和概率值

    提取概率排名前N个类的索引和概率值, N取决于num

    Args:
        distribution: 类别概率分布
                      list[]
        threshold: 概率阈值, 默认为None, 此时取最高概率
                   float32
        num: 结果类别数, 默认为1
             int
    Returns:
        res: 提取的类别索引和对应概率值
             list[(index, value), (), ...]
    """

    if num < 1:
        raise ValueError('Wrong value of category number!')

    idx_sorted = np.argsort(-distribution)
    if threshold is None:
        idx = idx_sorted[0]
        dist_ls = distribution.tolist()
        return [(idx, dist_ls[idx])]

    res = []
    for i in idx_sorted:
        dist = distribution.tolist()[i]
        if dist > threshold:
            res.append((i, dist))
        if len(res) >= num:
            break
    return res


def output_result_based_max(raw_image_path, samples_dir, save_path, labels_dict):

    """ 输出分类结果

    根据分类标签取概率值最大者对应类别，以tiff文件的格式输出分类结果

    Args:
        raw_image_path: 原始影像路径
                        str
        samples_dir: 与标签对应的样本目录
                     str
        save_path: 结果保存路径
                   str
        labels_dict: 与样本对应的分类标签
                     dict{'sample_name': [(index, value), ...]}
    """

    gdal.AllRegister()
    data_set = gdal.Open(raw_image_path)
    trans = data_set.GetGeoTransform()
    proj = data_set.GetProjection()
    img_rows, img_cols = data_set.RasterYSize, data_set.RasterXSize

    res_array = np.full((img_rows, img_cols), 255, dtype=np.uint8)
    value_array = np.full((img_rows, img_cols), -1, dtype=np.float32)

    g = os.walk(samples_dir)
    for path, dir_list, file_list in g:
        for file in file_list:
            area_id, ext = os.path.splitext(file)
            if ext != '.tif':
                continue
            if area_id not in labels_dict.keys():
                continue

            sys.stdout.write('\rProcessing %s ...' % area_id)
            sys.stdout.flush()

            cur_label, cur_value = labels_dict[area_id][0]
            file_path = os.path.join(path, file)
            cur_data_set = gdal.Open(file_path)
            cur_rows, cur_cols = cur_data_set.RasterYSize, cur_data_set.RasterXSize
            geo_x_org, geo_y_org = transform.imagexy2geo(cur_data_set, 0, 0)
            geo_x_end, geo_y_end = transform.imagexy2geo(cur_data_set, cur_rows-1, cur_cols-1)
            res_row_org, res_col_org = transform.geo2imagexy(data_set, geo_x_org, geo_y_org)
            res_row_end, res_col_end = transform.geo2imagexy(data_set, geo_x_end, geo_y_end)

            rect_arr = res_array[res_row_org:res_row_end, res_col_org:res_col_end]
            rect_value_arr = value_array[res_row_org:res_row_end, res_col_org:res_col_end]
            cond = rect_value_arr < cur_value
            rect_arr[cond] = cur_label
            rect_value_arr[cond] = cur_value

    save_tiff(save_path, res_array[np.newaxis, :], img_rows, img_cols, 1, trans, proj)


def output_result(raw_image_path, samples_info_path, labels_dict_path, save_path, category_index=-1):

    """

    保存各类概率值至图片文件中

    Args:
        raw_image_path: 原始影像路径
                        str
        samples_info_path: 样本信息路径，样本信息保存样本的坐标范围
                            str
        labels_dict_path: 概率标签路径
                          str
        save_path: 输出图片保存路径，要求扩展名为tif
                   str
        category_index: 指定待保存的类别索引，如果为-1，则保存所有类别
                        int

    """

    gdal.AllRegister()
    data_set = gdal.Open(raw_image_path)
    trans = data_set.GetGeoTransform()
    proj = data_set.GetProjection()
    img_rows, img_cols = data_set.RasterYSize, data_set.RasterXSize

    with open(samples_info_path, 'r') as s_f:
        samples_info = json.load(s_f)
    with open(labels_dict_path, 'r') as l_f:
        labels_dict = json.load(l_f)

    if category_index == -1:
        raster_num = len(list(labels_dict.values())[0])
        idx_list = list(range(0, raster_num))
    elif category_index < len(list(labels_dict.values())[0]):
        raster_num = 1
        idx_list = [category_index]
    else:
        raise ValueError('Wrong category index: %d' % category_index)
    res_array = np.zeros((raster_num, img_rows, img_cols), dtype=np.float32)
    for i in idx_list:
        for sample_name, labels in labels_dict.items():
            print('Processing %d-%s' % (i, sample_name))
            cur_value = labels[i]
            lon_nw, lat_nw, lon_se, lat_se = samples_info[sample_name].split(',')
            row_org, col_org = transform.lonlat2imagexy(data_set, float(lon_nw), float(lat_nw))
            row_end, col_end = transform.lonlat2imagexy(data_set, float(lon_se), float(lat_se))
            rect_arr = res_array[i, row_org:row_end+1, col_org:col_end+1]
            rect_arr[rect_arr < cur_value] = cur_value
    save_tiff(save_path, res_array, img_rows, img_cols, raster_num, trans, proj)


def flip_augment(samples_dir, label_path, label_save_path):

    """ 影像样本扩充

    翻转扩充影像样本

    Args:
        samples_dir: 样本目录
                     str
        label_path: 标签路径
                    str
        label_save_path: 标签保存路径
                         str

    """

    with open(label_path, 'r') as d_f:
        raw_label_dict = json.load(d_f)
    labels = {}
    for ar_id, v in raw_label_dict.items():
        labels[ar_id] = extract_category(np.array(v))

    augmented_labels = {}
    g = os.walk(samples_dir)
    for path, dir_list, file_list in g:
        for file in file_list:
            file_idx, ext = os.path.splitext(file)
            if ext != '.tif' or file_idx not in labels.keys():
                continue
            dist = labels[file_idx]

            # TODO: 设置跳过类型
            if dist[0][0] != 2 and dist[0][0] != 3:
                continue

            sys.stdout.write('\rProcessing %s ...' % file_idx)
            sys.stdout.flush()
            file_path = os.path.join(path, file)
            data_set = gdal.Open(file_path)
            for i in range(2):
                img_arr = data_set.ReadAsArray()
                r_num, height, width = img_arr.shape
                new_img = flip(img_arr, i)
                new_idx = '%s_1%d' % (file_idx, i)
                save_path = os.path.join(path, '%s.tif' % new_idx)
                save_tiff(save_path, new_img, width, height, r_num)
                augmented_labels[new_idx] = raw_label_dict[file_idx]
                if '_' in file_idx:
                    break
    raw_label_dict.update(augmented_labels)

    with open(label_save_path, 'w') as d_f:
        json.dump(raw_label_dict, d_f)


def rotate_augment(samples_dir, label_path, label_save_path):

    """ 影像样本扩充

    旋转扩充影像样本

    Args:
        samples_dir: 样本目录
                     str
        label_path: 标签路径
                    str
        label_save_path: 标签保存路径
                         str

    """

    with open(label_path, 'r') as d_f:
        raw_label_dict = json.load(d_f)
    labels = {}
    for ar_id, v in raw_label_dict.items():
        labels[ar_id] = extract_category(np.array(v))

    augmented_labels = {}
    g = os.walk(samples_dir)
    for path, dir_list, file_list in g:
        for file in file_list:
            file_idx, ext = os.path.splitext(file)
            if ext != '.tif' or file_idx not in labels.keys():
                continue
            dist = labels[file_idx]

            # TODO: 设置跳过类型
            if dist[0][0] != 2 and dist[0][0] != 3 and dist[0][0] != 0:
                continue

            sys.stdout.write('\rProcessing %s ...' % file_idx)
            sys.stdout.flush()
            file_path = os.path.join(path, file)
            data_set = gdal.Open(file_path)
            for i in range(3):
                img_arr = data_set.ReadAsArray()
                r_num, height, width = img_arr.shape
                new_img = rotate(img_arr, (i+1)*90)
                new_idx = '%s_0%d' % (file_idx, i)
                save_path = os.path.join(path, '%s.tif' % new_idx)
                save_tiff(save_path, new_img, width, height, r_num)
                augmented_labels[new_idx] = raw_label_dict[file_idx]
    raw_label_dict.update(augmented_labels)
    with open(label_save_path, 'w') as d_f:
        json.dump(raw_label_dict, d_f)


def record_augment(label_path, label_save_path, augment_code=0, target_index=None):

    """
    将样本扩充记录保存至文件中

    Args:
        label_path: 原始标签路径
                    str
        label_save_path: 标签保存路径
                         str
        augment_code: 扩充方式，默认为0即旋转扩充
                      int 0 or 1
        target_index: 特定的扩充类别索引列表，默认为None
                      list
    """

    with open(label_path, 'r') as d_f:
        raw_label_dict = json.load(d_f)
    labels = {}
    for ar_id, v in raw_label_dict.items():
        labels[ar_id] = extract_category(np.array(v))

    augmented_labels = {}
    for file_idx in labels.keys():
        dist = labels[file_idx]
        # TODO: 设置跳过类型
        if target_index and dist[0][0] in target_index:
            for i in range(1):
                new_idx = '%s_%d%d' % (file_idx, augment_code, i)
                augmented_labels[new_idx] = raw_label_dict[file_idx]
    raw_label_dict.update(augmented_labels)
    with open(label_save_path, 'w') as d_f:
        json.dump(raw_label_dict, d_f)


def save_tiff(save_path, img_data, rows, cols, raster_num, geotran_info=None, proj_info=None):

    """ tiff文件保存

    Args:
        save_path: 保存路径
                   str
        img_data: 待保存影像数据
                  array([[[]]])
        rows: 影像行数
              int
        cols: 影像列数
              int
        raster_num: 影像波段数
                    int
        geotran_info: 仿射变换参数
                      list or tuple ( , , , , , )
        proj_info: 投影信息
                   str
    """

    gdal.AllRegister()
    if 'int8' in img_data.dtype.name:
        data_type = gdal.GDT_Byte
    elif 'int16' in img_data.dtype.name:
        data_type = gdal.GDT_UInt16
    elif 'int32' in img_data.dtype.name:
        data_type = gdal.GDT_UInt32
    else:
        data_type = gdal.GDT_Float32
    driver = gdal.GetDriverByName('GTiff')
    data_set = driver.Create(save_path, int(cols), int(rows), int(raster_num), data_type)
    if data_set is None:
        raise ValueError('Wrong parameter of driver.Create().')
    elif geotran_info is not None and proj_info is not None:
        data_set.SetGeoTransform(geotran_info)
        data_set.SetProjection(proj_info)
    for idx in range(raster_num):
        data_set.GetRasterBand(idx + 1).WriteArray(img_data[idx])
    del data_set


def generate_topic_map(topic_dict, topic_num, size):

    """
    生成LDA主题分布栅格图

    Args:
        topic_dict: 主题分布字典
                    dict{"1": {"box": [0, 0, 256, 256],
                               "poi_sta": [3, 1, ...],
                               "label": [0, 0.07755155861377716, ...]},
                        ...}
        topic_num: 主题数量
        size: 栅格图大小
    Returns:
        res: 结果栅格数组
             array
    """

    shp = (size[0], size[1], topic_num)
    res = np.zeros(shp, dtype=np.float16)
    for k, v in topic_dict.items():
        box = v['box']
        topics = v['label']
        box_size = (box[2]-box[0], box[3]-box[1])
        dist_arr = np.array(topics).reshape(1, 1, topic_num)
        dist_arr = np.repeat(np.repeat(dist_arr, box_size[0], axis=0), box_size[1], axis=1)
        cur_org_arr = res[box[0]:box[2], box[1]:box[3], :]
        cur_org_arr = window_padding(dist_arr, cur_org_arr, 'max')

    return res


def flip(image_data, flip_code):
    if image_data.dtype is not np.uint8:
        image_data = bit.convert_to_uint8(image_data)
    return cv2.flip(image_data, flip_code)


def rotate(image_data, angle):
    if image_data.dtype is not np.uint8:
        image_data = bit.convert_to_uint8(image_data)
    n, rows, cols = image_data.shape
    res = np.zeros((n, rows, cols))
    for i in range(n):
        mat = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        res[i] = cv2.warpAffine(image_data[i], mat, (cols, rows))
    return res


def window_padding(new_arr, org_arr, method='max'):

    """ 窗口填充 """

    if new_arr.shape != org_arr.shape:
        raise IndexError('Wrong size.')

    res_arr = org_arr
    if method=='max':
        res_arr[new_arr>org_arr] = new_arr[new_arr>org_arr]
    elif method == 'min':
        res_arr[new_arr<org_arr] = new_arr[new_arr<org_arr]
    elif method == 'mean':
        res_arr += new_arr
        res_arr /= 2
    else:
        raise ValueError('Wrong method.')

    return res_arr


if __name__ == '__main__':

    gdal.AllRegister()
    img_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\study_area\study_area.tif'
    sample_img_dir = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\regular'
    old_label_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\label_files\regular_labels_300_100.json'
    new_label_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\label_files\augmented_labels_300_100.json'
    rect_info_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\regular_100_50\rect_dict_100_50.json'

    # record_augment(old_label_path, new_label_path, augment_code=0, target_index=[2, 3, 5])
    record_augment(new_label_path, new_label_path, augment_code=1, target_index=[0, 2, 3, 4])
    # output_classification_result(img_path, rect_info_path, old_label_path, r'result1.tif')
