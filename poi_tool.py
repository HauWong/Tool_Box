# !/usr/bin/env python
# -*- coding:utf-8 -*-

import math

import xlrd
import numpy as np
from scipy.interpolate import griddata

import transform as transform


def read(filename, total=False):

    """ 从Excel文件中读取poi数据 """

    workbook = xlrd.open_workbook(filename)
    if total:
        res = {}
        types = []
        i = 0
        for sheet in workbook.sheets():
            types.append(sheet.name)
            rows = sheet.nrows
            if rows <= 1:
                continue
            loc_arr = np.zeros((rows-1, 2))
            for row in range(1, rows):
                loc_arr[row-1, 0] = sheet.row_values(row)[1]
                loc_arr[row-1, 1] = sheet.row_values(row)[2]
            res[sheet.name] = loc_arr
            i += 1
        return types, res
    else:
        book_sheet = workbook.sheets()[0]
        rows = book_sheet.nrows
        res_arr = np.zeros((rows-1, 2))
        for row in range(1, rows):
            res_arr[row-1, 0] = book_sheet.row_values(row)[1]
            res_arr[row-1, 1] = book_sheet.row_values(row)[2]

        return res_arr


def coord_to_grid(dataset, location_array):

    """ 地理坐标转栅格坐标

    Args：
        dataset: gdal获取的影像数据集
        location_array: 待转换的地理坐标列表
                        array[[lng, lat],
                              ...
                              []]
    Return:
        res_arr: shape与location_array对应的栅格坐标列表
                 array[[raw, col],
                        ...
                        []]
    """

    poi_num = location_array.shape[0]
    res_arr = np.zeros((poi_num, 2), dtype='int16')
    for i in range(0, poi_num):
        lon = location_array[i, 0]
        lat = location_array[i, 1]
        geo_x, geo_y = transform.lonlat2geo(dataset, lon, lat)
        res_arr[i] = transform.geo2imagexy(dataset, geo_x, geo_y)
    return res_arr


def grid_to_coord(dataset, location_array):

    """ 栅格坐标转地理坐标

    Args：
        dataset: gdal读取的影像数据集
        location_array: 待转换的栅格坐标列表
                        array[[raw, col],
                              ...
                              []]
    Return:
        res_arr: shape与location_array对应的地理坐标列表
                 array[[lng, lat],
                       ...
                       []]
    """

    poi_num = location_array.shape[0]
    res_arr = np.zeros((poi_num, 2), dtype='float32')
    for i in range(0, poi_num):
        row = location_array[i, 0]
        col = location_array[i, 1]
        geo_x, geo_y = transform.imagexy2geo(dataset, row, col)
        res_arr[i] = transform.geo2lonlat(dataset, geo_x, geo_y)
    return res_arr


def cell_statistic(poi_array, region_rect, cell_size=1):

    """ 单元统计

    用于统计每个单元POI数量，进行栅格化
    参数cell_size默认为1，此时为保持分辨率的栅格化

    """

    rows = math.ceil((region_rect[2]-region_rect[0])/cell_size)
    cols = math.ceil((region_rect[3]-region_rect[1])/cell_size)
    res_arr = np.zeros((rows, cols), dtype='uint32')

    for poi in poi_array:
        if ((poi[0] >= region_rect[0]) and (poi[0] < region_rect[2])
                and (poi[1] >= region_rect[1]) and (poi[1] < region_rect[3])):
            x = math.floor(poi[0]/cell_size)
            y = math.floor(poi[1]/cell_size)
            res_arr[x, y] += 1

    return res_arr


def box_statistic(poi_array, box_size, step, img_arr):

    """ 带步长的块统计

    用于滑动重叠窗口对研究区域的POI数量进行分块统计

    """

    region_rect = poi_array.shape
    rows = math.floor((region_rect[0]-box_size)/step+1)
    cols = math.floor((region_rect[1]-box_size)/step+1)
    res_arr = np.zeros((rows, cols), dtype='uint16')

    for i in range(0, region_rect[0], step):
        for j in range(0, region_rect[1], step):
            if i+box_size <= region_rect[0] and j+box_size <= region_rect[1]:
                if count_percent(img_arr[i:i+box_size, j:j+box_size], 0) > 0.8:
                    continue
                cur_box = poi_array[i:i+box_size, j:j+box_size].copy()
                r_idx = int(i/step)
                c_idx = int(j/step)
                res_arr[r_idx, c_idx] = cur_box.sum()
            else:
                continue
    return res_arr


def expand(poi_array, region_rect, expand_width=21, gap=5):
    res_arr = poi_array.copy()
    for poi in poi_array:
        center_x = poi[0]
        center_y = poi[1]
        radius = int((expand_width-1)/2)
        for i in range(-radius, radius+1, gap):
            for j in range(-radius, radius+1, gap):
                cur_p_x = center_x+i
                cur_p_y = center_y+j
                if ((cur_p_x <= region_rect[0]) or (cur_p_x >= region_rect[2])
                        or (cur_p_y <= region_rect[1]) or (cur_p_y >= region_rect[3])):
                    continue
                res_arr = np.insert(res_arr, 0, (cur_p_x, cur_p_y), axis=0)
    return res_arr


def extract_pois(pois, rect_loc):
    res = {}
    lng_min = rect_loc[0]
    lng_max = rect_loc[2]
    lat_min = rect_loc[3]
    lat_max = rect_loc[1]
    for poi_type in pois.keys():
        cur_poi = pois[poi_type]
        cond = (cur_poi[:, 0] >= lng_min) & (cur_poi[:, 0] < lng_max) & \
               (cur_poi[:, 1] >= lat_min) & (cur_poi[:, 1] < lat_max)
        res[poi_type] = cur_poi[cond]

    return res


def idw_interpolation(poi_array, region_rect, influence_radius, rounding=0):

    """ 反距离权重插值 """

    def distance(point1, point2):
        tmp = math.pow((point1[0]-point2[0]), 2)+math.pow((point1[1]-point2[1]), 2)
        dist_res = math.sqrt(tmp)
        return dist_res

    def calculate_inter_value(point):
        sum0 = 0
        sum1 = 0
        for r in range(-influence_radius, influence_radius+1):
            for c in range(-influence_radius, influence_radius+1):
                if r == 0 and c == 0:
                    continue
                raw_idx = point[0]+r
                col_idx = point[1]+c
                if ((raw_idx < region_rect[0] or raw_idx >= region_rect[2]) or
                        (col_idx < region_rect[1] or col_idx >= region_rect[3])):
                    continue
                else:
                    cur_dis = distance(point, (raw_idx, col_idx))
                    sum0 += poi_array[raw_idx, col_idx]/cur_dis
                    sum1 += 1/cur_dis
        value_res = sum0 / sum1
        return value_res

    res_array = poi_array.copy()
    for i in range(region_rect[0], region_rect[2]):
        for j in range(region_rect[1], region_rect[3]):
            if poi_array[i, j] != 0:
                continue
            else:
                res_array[i, j] = calculate_inter_value((i, j))
    if rounding:
        res_array = np.floor(res_array*rounding)
    return res_array


def grid_interpolation(poi_array):

    """ 使用scipy插值方法进行三次样条插值 """

    def get_points(cond):
        point_idx_arr = np.where(cond)
        point_num = len(point_idx_arr[0])
        point_arr = np.zeros((point_num, 2), dtype='uint32')
        point_arr[:, 0] = point_idx_arr[0]
        point_arr[:, 1] = point_idx_arr[1]
        return point_arr

    points = get_points(poi_array != 0)
    point_values = poi_array[poi_array != 0]
    xi = get_points(poi_array == 0)
    zs = griddata(points, point_values, xi, method='cubic', fill_value=0)
    res_arr = poi_array.copy().astype('float32')
    for i in range(0, len(zs)):
        r_idx = xi[i][0]
        c_idx = xi[i][1]
        res_arr[r_idx, c_idx] = zs[i]
    return res_arr


def count_percent(arr, value):

    """ 计算某像素值数目的百分比 """

    cond = arr == value
    value_arr = np.extract(cond, arr)
    percent = float(value_arr.size)/float(arr.size)
    return percent


def convert_grid(dataset, poi_dict, box_boundary, types):

    """
    Convert poi locations in target box into a grid-format
    :param dataset: gdal dataset, must have geo-proj
    :param poi_dict: dict, dictionary of pois
    :param box_boundary: tuple, boundary of target box
    :param types: list, classes list of pois above
    :return: array result
    """

    channel_num = len(poi_dict)
    if len(types) != channel_num:
        print('Warning: Types number unmatch.')

    h = box_boundary[2] - box_boundary[0]
    w = box_boundary[3] - box_boundary[1]
    res_shape = (h, w, len(types))
    res = np.zeros(res_shape, dtype=np.int8)
    for i in range(len(types)):
        if types[i] not in poi_dict.keys():
            print('Warning: "Type" %s not found.' % types[i])
            continue
        poi_list = coord_to_grid(dataset, poi_dict[types[i]])
        print(poi_list.min())
        print(poi_list.max())
        for poi in poi_list:
            row, col = poi
            if row >= box_boundary[0] and \
               row < box_boundary[2] and \
               col >= box_boundary[1] and \
               col < box_boundary[3]:
                res[row, col, i] += 1
    return res


if __name__=='__main__':

    '''--------------生成栅格文件----------------'''

    img_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\regular\00002.tif'
    poi_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\regular_pois\00002\total.xlsx'
    type_ls = ['administrative', 'amusement', 'educational', 'market', 'residential', 'service', 'medical']
    type_ls = ['%s_00' % t for t in type_ls]

    from osgeo import gdal
    gdal.AllRegister()
    ds = gdal.Open(img_path)
    _, poi_d = read(poi_path, True)
    print(poi_d.keys())
    poi_grid = convert_grid(ds, poi_d, [0, 0, 227, 227], type_ls)
    print(poi_grid.shape)