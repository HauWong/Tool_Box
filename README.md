# 个人实验使用工具箱（\*_\*)

## 基本的图片处理工具
*basic_image_tool.py*  
包括
1. `convert_to_uint8(img_arr)`  
    将图像转换为8-bit格式  
2. `normalize(arr)`  
    最值归一化
3. `statistics(arr, num)`  
    统计直方图
4. `dms_to_ten(loc_dms)`  
    将度分秒格式的坐标转换为十进制
5. `ten_to_dms(loc_ten)`  
    将十进制格式的坐标转换为度分秒  
6. `count_percent(arr, value)`  
    计算某数值在图像中的百分比
7. `flip(img_arr, flip_code=0)`  
    利用opencv提供的工具进行图像翻转

## 凸包识别工具
*generate_convex_hull.py*  
包括  
1. `calculate_convex_hull(points)`   
    根据提供二值图像的角点坐标提取凸包
```python
    Args:
        points: 待提取凸包二值图像的角点列表，
                list([row1, col1], [row2, col2], ...)
    Returns:
         res: 凸包角点列表
              list([row1, row2], ...)
```

## ROI提取工具
*roi_extract.py*  
包括
1. `extract_roi(image_path, xml_file_path, save_dir)`  
    可实现从包含多个ROI的.xml文件中自动提取所有ROI的所有区域，并保存至.tif文件中
```python  
   Args:
        image_path: 带有地理信息的原始影像文件路径
                  str
        xml_file_path: 带含多个ROI且具有与img_path文件相同地理参考信息的.xml文件路径
                       str
        save_dir: 保存文件目录
                   str
```
2. `extract_roi_box(image_path, xml_file_path, save_dir)`  
    可实现从包含多个ROI的.xml文件中自动提取所有ROI的所有区域的最小外接矩形，并保存至.tif文件中
```python
    Args:
        image_path: 带有地理信息的原始影像文件路径
                str
        xml_file_path: 带含多个ROI且具有与img_path文件相同地理参考信息的.xml文件路径
                    str
        save_dir: 保存文件目录
                str
    Returns:
        res: 外接矩形列表
```
3. `extract_lonlat_str(image_path, xml_file_path, save_path)`  
        从xml文件中提取各块的角点坐标, 并保存至文件
```python
    Args:
        image_path: 带有地理信息的原始影像文件路径
                  str
        xml_file_path: 带含多个ROI且具有与img_path文件相同地理参考信息的.xml文件路径
                       str
        save_path: 角点坐标文件保存路径，后缀为.json
                   str
    Returns:
        res_dict: 从xml文件中提取的各块角点经纬度坐标字典
                  dict{'region_name': ['lon_1,lat_1|lon_2, lat_2|lon_3,lat_3', ...], ...}
```
4. `mask_roi(image_path, mask_path, save_dir)`  
可实现从包含多个ROI的.tif文件中自动提取所有ROI的所有区域，并将每块影像保存至.tif文件中, ROI多边形dict保存至.json文件中, 保存每个块的面积保存至area.json中
```python
    Args:
        image_path: 带有地理信息的原始影像文件路径
                  str
        mask_path: 与影像地理信息相同的tif格式的掩膜文件路径
                   str
        save_dir: 保存文件目录
                  str
```
5. `clip_with_centroids(image_path, save_dir, centroids_dict, shape)`  
    以给定点为中心裁剪固定大小(取决于shape)的区域得到ROI, 并保存为.tif文件
```python
    Args:
        image_path: 原始影像路径
                    str
        save_dir: ROI影像保存目录
                  str
        centroids_dict: 中心点字典
                        dict{'polygon_name': array([lon, lat]), ...}
        shape: 裁剪ROI大小
               tuple(height, width)
```
6. `clip_with_sliding_window(image_path, save_dir, shape, step, save_img=True)`  
    以固定大小的窗口、以固定步长在影像中滑动，截取ROI
```python
    Args:
        image_path: 原始影像路径
                    str
        save_dir: ROI保存目录
                  str
        shape: 窗口大小
               tuple(height, width)
        step: 步长
              int
        save_img: 保存图像
                  bool
```
7. `clip(path)`  
    裁剪影像至指定大小
```python
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
```
8. `array2vector(array)`  
    根据实心二值数组提取凸包
```python
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
```
9. `paint_rectangles(image, box_ls, color=(255, 0, 0), width=2)`  
    在图像上批量绘制矩形框
```python
    Args：
        image: array, 图像数组, shape=(w, h, c)
        box_ls: list, 矩形框列表, [(row_min, row_max, col_min, col_max), ...]
        color: tuple, 8-bit RGB形式的框体颜色
        width: int, 框线宽度
    Returns：
        image: array, 结果图像
```
## 图片输出工具
*output.py*  
包括
1. `extract_category(distribution, threshold=None, num=1)`
    提取概率排名前N个类的索引和概率值, N取决于num
```python
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
```
2. `output_result_based_max(raw_image_path, samples_dir, save_path, labels_dict)`  
    根据分类标签取概率值最大者对应类别，以tiff文件的格式输出分类结果
```python
    Args:
        raw_image_path: 原始影像路径
                        str
        samples_dir: 与标签对应的样本目录
                     str
        save_path: 结果保存路径
                   str
        labels_dict: 与样本对应的分类标签
                     dict{'sample_name': [(index, value), ...]}
```
3. `output_result(raw_image_path, samples_info_path, labels_dict_path, save_path, category_index=-1)`  
    保存各类概率值至图片文件中
```python
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
```
4. `flip_augment(samples_dir, label_path, label_save_path)`  
翻转扩充影像样本
```python
    Args:
        samples_dir: 样本目录
                     str
        label_path: 标签路径
                    str
        label_save_path: 标签保存路径
                         str
```
5. `rotate_augment(samples_dir, label_path, label_save_path)`  
    旋转扩充影像样本
```python
    Args:
        samples_dir: 样本目录
                     str
        label_path: 标签路径
                    str
        label_save_path: 标签保存路径
                         str
```   
6. `def save_tiff(save_path, img_data, rows, cols, raster_num, geotran_info=None, proj_info=None)`  
    tiff文件保存
```python
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
```   
7. `generate_topic_map(topic_dict, topic_num, size)`
    生成LDA主题分布栅格图
```python
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
```
## 坐标转换工具
*transform.py*  
包括
1. `geo2lonlat(dataset, x, y)`  
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
```python    
    Args:
        dataset: GDAL地理数据
        x: 投影坐标x
        y: 投影坐标y
    Returns:
        coords[:2]: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
```
2. `lonlat2geo(dataset, lon, lat)`  
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
```python
    Args:
        dataset: GDAL地理数据
        lon: 地理坐标lon经度
        lat: 地理坐标lat纬度
    Returns:
        coords[:2]: 经纬度坐标(lon, lat)对应的投影坐标
```
3. `imagexy2geo(dataset, row, col)`  
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
```python
    Args:
        dataset: GDAL地理数据
        row: 像素的行号
        col: 像素的列号
    Returns:
        px, py: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
```
4. `geo2imagexy(dataset, x, y)`
    根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
```python
    Args:
        dataset: GDAL地理数据
        x: 投影或地理坐标x
        y: 投影或地理坐标y
    Returns:
        row, col: 投影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
```
5. `lonlat2imagexy(dataset, lon, lat)`
    将经纬度坐标转为影像图上坐标（行列号）
```python
    Args:
        dataset: GDAL地理数据
        lon: 地理坐标lon经度
        lat: 地理坐标lat纬度
    Returns:
        row, col: 地理坐标(lon, lat)对应的影像图上行列号(row, col)
```
6. `imagexy2lonlat(dataset, row, col)`
    根据GDAL的六参数模型将影像图上坐标（行列号）转为地理坐标
```python
    Args:
        dataset: GDAL地理数据
        row: 像素的行号
        col: 像素的列号
    Returns:
        lon, lat: 经纬度坐标(lon, lat)对应的投影坐标
```
## POI基本处理工具
*poi_tool.py*  
包括
1. `read(filename, total=False)`  
    从Excel文件中读取poi数据，要求*.xlsx文件格式内第一行为表头，从第二行开始为每个POI的属性值，第二列为longitude，第三列为latitude
```python
    Args:
        filename: 文件名
                  str
        total: 是否从所有工作表中读取
               bool
    Returns:
        types: 工作表名列表，when total==True
               list
        res: 读取结果，when total==True
             dict
        res_arr: 读取结果，when total==False
                 array
```
2. `coord_to_grid(dataset, location_array)`  
    将所有POI的地理坐标转为栅格坐标
```python
    Args：
        dataset: gdal获取的影像数据集
        location_array: 待转换的地理坐标列表
                        array[[lng, lat],
                              ...
                              []]
    Returns:
        res_arr: shape与location_array对应的栅格坐标列表
                 array[[raw, col],
                        ...
                        []]
```
3.  `grid_to_coord(dataset, location_array)`  
    栅格坐标转地理坐标
```python
    Args：
        dataset: gdal读取的影像数据集
        location_array: 待转换的栅格坐标列表
                        array[[raw, col],
                              ...
                              []]
    Returns:
        res_arr: shape与location_array对应的地理坐标列表
                 array[[lng, lat],
                       ...
                       []]
```
4. `cell_count(poi_array, region_rect, cell_size=1)`  
    用于统计每个单元POI数量，进行栅格化
    参数cell_size默认为1，此时为保持分辨率的栅格化
```python
    Args：
        poi_array: poi数组
                   array
        region_rect: 目标区域的矩形边界
                     tuple(row0, col0, row1, col1)                      
    Returns:
        res_arr: 统计结果栅格
                 array
```
5. `box_count(poi_array, box_size, step, img_arr=None)`  
    用于滑动重叠窗口对研究区域的POI数量进行分块统计
```python
    Args：
        poi_array: poi数组
                   array
        box_size: 滑动窗口的大小
                  int
        step: 滑动步长
              int
        img_arr: 参考影像
                 array                       
    Returns:
        res_arr: 统计结果数组
                 array
```
6. `convert_grid(dataset, poi_dict, box_boundary, types)`  
    将目标区域内的POI点投影到栅格数组上
```python
    Args:
        dataset: GDAL地理数据
                 gdal Dataset
        poi_dict: 包含多个类别的POI字典
                  dict
        box_boundary: 目标区域的边界框
                      tuple
        types: 上述POI的类别列表
               list
    Returns:
        res: 栅格数组
             array
```