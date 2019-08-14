# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from osgeo import gdal
from osgeo import osr
import numpy as np

gdal.AllRegister()


def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    Args:
        dataset: GDAL地理数据
    Returns:
        prosrs, geosrs: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2lonlat(dataset, x, y):
    """
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    Args:
        dataset: GDAL地理数据
        x: 投影坐标x
        y: 投影坐标y
    Returns:
        coords[:2]: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    """
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def lonlat2geo(dataset, lon, lat):
    """
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    Args:
        dataset: GDAL地理数据
        lon: 地理坐标lon经度
        lat: 地理坐标lat纬度
    Returns:
        coords[:2]: 经纬度坐标(lon, lat)对应的投影坐标
    """
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def imagexy2geo(dataset, row, col):
    """
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    Args:
        dataset: GDAL地理数据
        row: 像素的行号
        col: 像素的列号
    Returns:
        px, py: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    """
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def geo2imagexy(dataset, x, y):
    """
    根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    Args:
        dataset: GDAL地理数据
        x: 投影或地理坐标x
        y: 投影或地理坐标y
    Returns:
        row, col: 投影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    """
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    col, row = np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解
    return int(row), int(col)


def lonlat2imagexy(dataset, lon, lat):
    """
    将经纬度坐标转为影像图上坐标（行列号）
    Args:
        dataset: GDAL地理数据
        lon: 地理坐标lon经度
        lat: 地理坐标lat纬度
    Returns:
        row, col: 地理坐标(lon, lat)对应的影像图上行列号(row, col)
    """
    geo_x, geo_y = lonlat2geo(dataset, lon, lat)
    row, col = geo2imagexy(dataset, geo_x, geo_y)
    return int(round(row)), int(round(col))


def imagexy2lonlat(dataset, row, col):
    """
    根据GDAL的六参数模型将影像图上坐标（行列号）转为地理坐标
    Args:
        dataset: GDAL地理数据
        row: 像素的行号
        col: 像素的列号
    Returns:
        lon, lat: 经纬度坐标(lon, lat)对应的投影坐标

    """

    geo_x, geo_y = imagexy2geo(dataset, row, col)
    lon, lat = geo2lonlat(dataset, geo_x, geo_y)
    return lon, lat


if __name__ == '__main__':
    gdal.AllRegister()
    save_path = r'D:\Documents\Study\Python\POI_Spider\assistants\study_area_locations.txt'
    img_path = r'D:\Documents\Study\Python\Self_Supervision\data\study_area\study_area.tif'
    with open(save_path, 'w') as loc_f:
        data_set = gdal.Open(img_path)
        row = data_set.RasterYSize
        col = data_set.RasterXSize
        nw_geox, nw_geoy = imagexy2geo(data_set, 0, 0)
        nw_lng, nw_lat = geo2lonlat(data_set, nw_geox, nw_geoy)
        se_geox, se_geoy = imagexy2geo(data_set, row, col)
        se_lng, se_lat = geo2lonlat(data_set, se_geox, se_geoy)
        loc_str = '{\"area_01\": [\"%.6f,%.6f,%.6f,%.6f\"]}' % (nw_lng, nw_lat, se_lng, se_lat)
        loc_f.write(loc_str)


