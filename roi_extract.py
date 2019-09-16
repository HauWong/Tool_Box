# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

import json
import numpy as np
import cv2
from osgeo import gdal

import transform
import generate_convex_hull as gch
from output import save_tiff


def extract_roi(image_path, xml_file_path, save_dir):

    """ ROI提取

    可实现从包含多个ROI的.xml文件中自动提取所有ROI的所有区域，并保存至.tif文件中

    Args:
        image_path: 带有地理信息的原始影像文件路径
                  str
        xml_file_path: 带含多个ROI且具有与img_path文件相同地理参考信息的.xml文件路径
                       str
        save_dir: 保存文件目录
                   str
    """

    gdal.AllRegister()
    print('Reading raw image ...')
    data_set = gdal.Open(image_path)
    if data_set is None:
        raise FileNotFoundError('File %s Not found' % image_path)
    img_tran = data_set.GetGeoTransform()
    img_proj = data_set.GetProjection()
    img_data = data_set.ReadAsArray()
    raster_num, img_rows, img_cols = img_data.shape

    print('Reading .xml file ...')
    if xml_file_path[-4:] != '.xml':
        raise TypeError('Unrecognizable type: \'%s\'' % xml_file_path[-4:])
    xml_proj, regions_dict = extract_xml_info(xml_file_path)
    print('Image Projection: %s' % img_proj)
    print('ROI Projection: %s' % xml_proj)
    if 'PROJCS' in xml_proj:
        with_projcs = True
    else:
        with_projcs = False
    roi_dict = transform_geo_to_ctype(data_set, regions_dict, ctype='grid', with_projcs=with_projcs)

    for region_name in roi_dict.keys():
        plg_list = roi_dict[region_name]
        i = 0
        for plg in plg_list:
            i += 1
            sys.stdout.write('\rProcessing \"%s\" %04d ...' % (region_name, i))
            sys.stdout.flush()
            plg_points = np.array(plg[:-1], dtype='int32')
            rect = get_extent(plg_points)

            plg_row_min, plg_row_max, plg_col_min, plg_col_max = limit_boundary(rect, (0, img_rows, 0, img_cols))

            plg_points[:, 0] = plg_points[:, 0] - plg_row_min
            plg_points[:, 1] = plg_points[:, 1] - plg_col_min
            plg_points[:, [0, 1]] = plg_points[:, [1, 0]]
            plg_points = plg_points[np.newaxis]
            mask_rows = plg_row_max-plg_row_min
            mask_cols = plg_col_max-plg_col_min
            mask = np.zeros((mask_rows, mask_cols), dtype='uint8')
            cv2.polylines(mask, plg_points, 1, 255)
            cv2.fillPoly(mask, plg_points, 255)
            cond = mask == 0

            masked = img_data[:, plg_row_min:(plg_row_min+mask_rows), plg_col_min:(plg_col_min+mask_cols)].copy()
            for j in range(0, raster_num):
                masked[j][cond] = 0

            directory = os.path.join(save_dir, region_name)
            folder = os.path.exists(directory)
            if not folder:
                os.makedirs(directory)
            save_name = os.path.join(directory, '%04d.tif' % i)

            geo_x, geo_y = transform.imagexy2geo(data_set, plg_row_min, plg_col_min)
            masked_tran = list(img_tran)
            masked_tran[0] = geo_x
            masked_tran[3] = geo_y
            save_tiff(save_name, masked, mask_rows, mask_cols, raster_num, masked_tran, img_proj)
    del data_set
    print('\nMission completed!')


def extract_lonlat_str(image_path, xml_file_path, save_path):

    """ ROI角点坐标字符串提取

    从xml文件中提取各块的角点坐标, 并保存至文件

    Args:
        image_path: 带有地理信息的原始影像文件路径
                  str
        xml_file_path: 带含多个ROI且具有与img_path文件相同地理参考信息的.xml文件路径
                       str
        save_path: 角点坐标文件保存路径，后缀为.json
                   str
    Returns:
        res_dict: 从xml文件中提取的各块角点经纬度坐标字典
                  dict{'region_name': ['lon_1,lat_1|lon_2,lat_2|lon_3,lat_3', ...], ...}
    """

    gdal.AllRegister()
    data_set = gdal.Open(image_path)

    if xml_file_path[-4:] != '.xml':
        raise TypeError('Unrecognizable type: \'%s\'' % xml_file_path[-4:])
    proj, regions_dict = extract_xml_info(xml_file_path)
    if 'PROJCS' in proj:
        with_projcs = True
    else:
        with_projcs = False
    roi_dict = transform_geo_to_ctype(data_set, regions_dict, ctype='lonlat', with_projcs=with_projcs)

    res_dict = {}
    for region_name in roi_dict.keys():
        plg_list = roi_dict[region_name]
        plg_str_list = []
        for plg in plg_list:
            plg_str = ''
            for point in plg:
                plg_str += '%f,%f|' % (point[0], point[1])
            plg_str_list.append(plg_str[:-1])
        res_dict[region_name] = plg_str_list

    with open(save_path, 'w') as d_f:
        js = json.dumps(res_dict)
        d_f.write(js)

    return res_dict


def mask_roi(image_path, mask_path, save_dir):

    """ 从tif格式的掩膜文件中提取roi

    可实现从包含多个ROI的.tif文件中自动提取所有ROI的所有区域，
    并将每块影像保存至.tif文件中, ROI多边形dict保存至.json文件中
    保存每个块的面积保存至area.json中

    Args:
        image_path: 带有地理信息的原始影像文件路径
                  str
        mask_path: 与影像地理信息相同的tif格式的掩膜文件路径
                   str
        save_dir: 保存文件目录
                  str
    """

    gdal.AllRegister()
    print('Reading raw image ...')
    img_data_set = gdal.Open(image_path)
    print('Reading mask image ...')
    mask_data_set = gdal.Open(mask_path)
    if img_data_set is None:
        raise FileNotFoundError('File %s not found' % image_path)
    if mask_data_set is None:
        raise FileNotFoundError('File %s not found!' % mask_path)
    img_tran = img_data_set.GetGeoTransform()
    img_proj = img_data_set.GetProjection()
    mask_proj = mask_data_set.GetProjection()
    if img_proj != mask_proj:
        raise ValueError('Wrong mask file!')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_data = img_data_set.ReadAsArray()
    mask_data = mask_data_set.ReadAsArray()
    dict_save_name = os.path.join(save_dir, 'all_polygons.json')
    plg_str_list = []
    region_dict = {}
    area_dict = {}
    raster_num, img_rows, img_cols = img_data.shape
    rois_num = mask_data.max()
    for i in range(1, rois_num+1):
        sys.stdout.write('\rProcessing %dth polygon ...' % i)
        sys.stdout.flush()
        mask_idxs = np.argwhere(mask_data == i)
        if mask_idxs.shape[0] == 0:
            continue
        rect = get_extent(mask_idxs)
        plg_row_min, plg_row_max, plg_col_min, plg_col_max = limit_boundary(rect, (0, img_rows, 0, img_cols))
        mask_rows = plg_row_max - plg_row_min + 1
        mask_cols = plg_col_max - plg_col_min + 1
        mask = mask_data[plg_row_min:plg_row_max+1, plg_col_min:plg_col_max+1].copy()
        mask_flag = mask.copy()
        mask[mask_flag == i] = 1
        mask[mask_flag != i] = 0
        del mask_flag
        area = mask.sum()
        area_dict['%04d' % i] = int(area)

        # 提取并保存凸包
        plg = array2vector(mask)
        plg_str = ''
        for point in plg:
            p_geo_x, p_geo_y = transform.imagexy2geo(mask_data_set, plg_row_min+point[0], plg_col_min+point[1])
            p_lon, p_lat = transform.geo2lonlat(mask_data_set, p_geo_x, p_geo_y)
            plg_str += '%f,%f|' % (p_lon, p_lat)
        print(plg_str)
        print()
        plg_str_list.append(plg_str[:-1])
        region_dict['all_polygons'] = plg_str_list
        with open(dict_save_name, 'w') as d_f:
            js = json.dumps(region_dict)
            d_f.write(js)

        # 提取并保存影像
        masked = img_data[:, plg_row_min:(plg_row_min+mask_rows), plg_col_min:(plg_col_min+mask_cols)].copy()
        if masked.shape[1:3] != mask.shape:
            continue
        for j in range(0, raster_num):
            masked[j][mask != 1] = 0
        save_name = os.path.join(save_dir, '%04d.tif' % i)
        geo_x, geo_y = transform.imagexy2geo(img_data_set, plg_row_min, plg_col_min)
        masked_tran = list(img_tran)
        masked_tran[0] = geo_x
        masked_tran[3] = geo_y
        save_tiff(save_name, masked, mask_rows, mask_cols, raster_num, masked_tran, img_proj)
    area_path = os.path.join(save_dir, 'area.json')
    with open(area_path, 'w') as a_f:
        json.dump(area_dict, a_f)
    del mask_data_set
    del img_data_set
    print('\nMission completed!')


def clip_with_centroids(image_path, save_dir, centroids_dict, shape):

    """ 裁剪得到ROI

    以给定点为中心裁剪固定大小(取决于shape)的区域得到ROI, 并保存为.tif文件

    Args:
        image_path: 原始影像路径
                    str
        save_dir: ROI影像保存目录
                  str
        centroids_dict: 中心点字典
                        dict{'polygon_name': array([lon, lat]), ...}
        shape: 裁剪ROI大小
               tuple(height, width)
    """

    gdal.AllRegister()
    data_set = gdal.Open(image_path)
    if data_set is None:
        raise FileNotFoundError('File %s not found' % image_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_tran = data_set.GetGeoTransform()
    img_proj = data_set.GetProjection()
    img_data = data_set.ReadAsArray()
    raster_num, img_rows, img_cols = img_data.shape
    for plg_name, centroid in centroids_dict.items():
        geo_x, geo_y = transform.lonlat2geo(data_set, centroid[0], centroid[1])
        centroid_grid = transform.geo2imagexy(data_set, geo_x, geo_y)
        sys.stdout.write('\rProcessing centroid (%f, %f) --> (%d, %d) ...' %
              (centroid[0], centroid[1], centroid_grid[0], centroid_grid[1]))
        sys.stdout.flush()

        row_min = int(centroid_grid[0]-int(shape[0]/2))
        if row_min < 0:
            row_min = 0
        col_min = int(centroid_grid[1]-int(shape[1]/2))
        if col_min < 0:
            col_min = 0
        row_max = int(centroid_grid[0]+shape[0]-int(shape[0]/2))
        col_max = int(centroid_grid[1]+shape[1]-int(shape[1]/2))
        roi_arr = img_data[:, row_min:row_max, col_min:col_max].copy()
        roi_tran = list(img_tran)
        roi_tran[0], roi_tran[3] = transform.imagexy2geo(data_set, row_min, col_min)

        save_name = os.path.join(save_dir, '%s.tif' % plg_name)
        save_tiff(save_name, roi_arr, roi_arr.shape[1], roi_arr.shape[2], raster_num, roi_tran, img_proj)
    del data_set
    print('\nMission completed!')


def clip_with_sliding_window(image_path, save_dir, shape, step):

    """ 以滑动窗口截取ROI

    以固定大小的窗口、以固定步长在影像中滑动，截取ROI

    Args:
        image_path: 原始影像路径
                    str
        save_dir: ROI保存目录
                  str
        shape: 窗口大小
               tuple(height, width)
        step: 步长
              int
    """

    gdal.AllRegister()
    data_set = gdal.Open(image_path)
    if data_set is None:
        raise FileNotFoundError('File %s not found' % image_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_tran = data_set.GetGeoTransform()
    img_proj = data_set.GetProjection()
    img_data = data_set.ReadAsArray()
    rect_dict = {}
    raster_num, img_rows, img_cols = img_data.shape
    total_num = int((img_rows-shape[0])/step+1)*int((img_cols-shape[1])/step+1)
    roi_idx = 1
    for i in range(0, img_rows, step):
        for j in range(0, img_cols, step):
            roi_arr = img_data[:, i:i+shape[0], j:j+shape[1]].copy()
            if roi_arr.shape[1:3] != shape:
                continue
            sys.stdout.write('\rProcessing %d/%d ...' % (roi_idx, total_num))
            sys.stdout.flush()
            geo_x_nw, geo_y_nw = transform.imagexy2geo(data_set, i, j)
            lon_nw, lat_nw = transform.geo2lonlat(data_set, geo_x_nw, geo_y_nw)
            geo_x_se, geo_y_se = transform.imagexy2geo(data_set, i+shape[0], j+shape[1])
            lon_se, lat_se = transform.geo2lonlat(data_set, geo_x_se, geo_y_se)
            rect_str = '%0.6f,%0.6f,%0.6f,%0.6f' % (lon_nw, lat_nw, lon_se, lat_se)
            rect_dict['%05d' % roi_idx] = rect_str

            roi_tran = list(img_tran)
            roi_tran[0], roi_tran[3] = geo_x_nw, geo_y_nw
            save_name = os.path.join(save_dir, '%05d.tif' % roi_idx)
            # save_tiff(save_name, roi_arr, shape[0], shape[1], raster_num, roi_tran, img_proj)
            roi_idx += 1
    del data_set
    rect_save_name = os.path.join(save_dir, 'rect_dict_%d_%d.json' % (shape[0], step))
    with open(rect_save_name, 'w') as d_f:
        json.dump(rect_dict, d_f)


def get_extent(points):

    """ 获取多边形的最小外接矩形的四个角点 """

    row_min = points[:, 0].min()
    row_max = points[:, 0].max()
    col_min = points[:, 1].min()
    col_max = points[:, 1].max()
    return row_min, row_max, col_min, col_max


def limit_boundary(rect, boundary):

    """ 限制矩形边界 """

    res = list(rect)
    if rect[0] < boundary[0]:
        res[0] = boundary[0]
    if rect[1] > boundary[1]:
        res[1] = boundary[1]
    if rect[2] < boundary[2]:
        res[2] = boundary[2]
    if rect[3] > boundary[3]:
        res[3] = boundary[3]
    return tuple(res)


def extract_xml_info(path):

    """
    从.xml文件中提取ROI信息

    Args:
        path: .xml文件路径
              str
    Returns:
        proj: .xml的地理参考信息
              str
        regions_dict: 所有ROI的多边形信息
                      dict({'roi_name': ['x_1 y_1 x_2 y_2 x_3 y_3', '']})
    """

    name_pattern = re.compile(r'name="(.*)"\s')
    proj_pattern = re.compile(r'<CoordSysStr>(.*)</CoordSysStr>')
    coord_pattern = re.compile(r'<Coordinates>\n([\d*.\d*\s?]+)\n\s*</Coordinates>')

    regions_dict = {}
    with open(path, 'r') as roi_f:
        xml_info_str = roi_f.read()
        proj = re.findall(proj_pattern, xml_info_str)[0]
        region_list = xml_info_str.split(r'</Region>')
        for region_str in region_list:
            name = re.findall(name_pattern, region_str)
            if len(name) == 0:
                continue
            coord_str_list = re.findall(coord_pattern, region_str)
            regions_dict[name[0]] = coord_str_list
    return proj, regions_dict


def transform_geo_to_ctype(data_set, regions_dict, with_projcs, ctype='lonlat'):

    """
    将dict中的字符串形式的地理坐标转换为栅格坐标或经纬度坐标（取决于ctype）

    Args:
        data_set: 影像GDAL数据集
                  GDALDataset
        regions_dict: 所有ROI的多边形信息
                      dict{'region_name': ['x_1 y_1 x_2 y_2 x_3 y_3', ...]}
        with_projcs: .xml文件是否具有投影信息
                     Boolean
        ctype: 转换后的坐标类型, 两种选择: 'grid', 'lonlat', 默认为后者
               str
    Returns:
        res: 经过转换后，保存有多边形角点栅格坐标的dict
             dict({'roi_name': [[(row_1, col_1), (row_2, col_2), (row_3, col_3)], []]})
             或
             dict({'roi_name': [[(lon_1, lat_1), (lon_2, lon_2), (lon_3, lat_3)], []]})
    """
    res = {}
    for region_name in regions_dict.keys():
        cur_region = []
        str_list = regions_dict[region_name]
        for coords_str in str_list:
            plg = []
            coords = coords_str.split(' ')
            x_list = [float(_) for _ in coords[::2]]
            y_list = [float(_) for _ in coords[1::2]]
            for i in range(len(x_list)):
                if ctype == 'grid':
                    if with_projcs:
                        geo_x, geo_y = x_list[i], y_list[i]
                    else:
                        geo_x, geo_y = transform.lonlat2geo(data_set, x_list[i], y_list[i])
                    point = transform.geo2imagexy(data_set, geo_x, geo_y)
                    point = np.ceil(point)
                    point = point.astype('int16')
                elif ctype == 'lonlat':
                    if with_projcs:
                        point = transform.geo2lonlat(data_set, x_list[i], y_list[i])
                    else:
                        point = (x_list[i], y_list[i])
                else:
                    raise TypeError('Unrecognizable type: \'%s\'' % ctype)
                plg.append(tuple(point))
            cur_region.append(plg)
        res[region_name] = cur_region
    return res


def clip(image_path, save_path, x_size, y_size, offset_x=0, offset_y=0):

    """
    裁剪影像至指定大小

    Args:
        image_path: 原始影像路径
                    str
        save_path: 结果保存路径
                   str
        x_size: 行大小
                int
        y_size: 列大小
                int
        offset_x: 行偏移量，默认为0
                  int
        offset_y: 列偏移量，默认为0
                  int
    """

    gdal.AllRegister()
    print('Reading raw image ...')
    data_set = gdal.Open(image_path)
    if data_set is None:
        raise FileNotFoundError('File %s Not found' % image_path)
    img_tran = data_set.GetGeoTransform()
    img_proj = data_set.GetProjection()
    img_data = data_set.ReadAsArray()
    raster_num, img_rows, img_cols = img_data.shape

    if offset_x+x_size > img_rows:
        end_x = img_rows
    else:
        end_x = offset_x+x_size
    if offset_y+y_size > img_cols:
        end_y = img_cols
    else:
        end_y = offset_y+y_size

    res_data = img_data[:, offset_x:end_x, offset_y:end_y]
    save_tiff(save_path, res_data, end_x-offset_x, end_y-offset_y, raster_num, geotran_info=img_tran, proj_info=img_proj)


def array2vector(array):
    """
    根据实心二值数组提取凸包

    Args:
        array: 二维实心二值数组
               array([[0, 0, 1, 1, 0],
                      [0, 1, 1, 1, 1]
                      ...
                      []])
               shape(height, width)
    Returns:
        plg: 凸包多边形
             list[[1, 2], ..., [1, 2]]

    """

    h, w = array.shape
    corners = []
    for i in range(h):
        for j in range(w):
            win = np.zeros((3, 3), dtype='uint16')
            for i0 in range(-1, 2):
                for j0 in range(-1, 2):
                    if i + i0 < 0 or j + j0 < 0 or i + i0 >= h or j + j0 >= w:
                        continue
                    win[i0 + 1, j0 + 1] = array[i + i0, j + j0]
            if win[1, 1] == 1:
                if win.sum() <= 5 or (win.sum() >= 7 and 1 in np.argwhere(win == 0)):
                    corners.append([i, j])
    plg = gch.calculate_convex_hull(corners)
    return plg


if __name__ == '__main__':
    img_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\study_area\study_area.tif'
    msk_path = r'D:\Documents\Study\Python\Self_Supervision\data\jinniu\jinniudistrict\ground_truth_sorted.tif'
    xml_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\test.xml'
    save_path = r'D:\Documents\Study\Python\Self_Supervision\data\shijiazhuang\regular_100_50'
    centroids_path = r'D:\Documents\Study\Python\Self_Supervision\data\centroids.json'
    plg_save_name = r'D:\Documents\Study\Python\POI_Spider\assistants\regions_large.json'
    # with open(centroids_path, 'r') as c_f:
    #     centroids_d = json.load(c_f)
    # clip_with_centroids(img_path, save_path, centroids_d, (227, 227))
    # mask_roi(img_path, msk_path, save_path)
    # extract_roi(img_path, xml_path, save_path)
    # res_d = extract_lonlat_str(img_path, xml_path, plg_save_name)
    clip_with_sliding_window(img_path, save_path, (50, 50), 50)
    # print(res_d)
    # extract_lonlat_str(img_path, xml_path, save_path)
